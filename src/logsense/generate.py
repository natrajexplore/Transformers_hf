"""Phase 1: synthetic, anonymized, labeled syslog generator.

Produces log lines for Cisco IOS/IOS-XE, Cisco ISE, Cisco C9800 WLC, and PAN-OS, each labeled
with one category from configs/default.yaml, plus character spans for entities (IP, USER,
INTERFACE, MAC, RULE, VLAN) that Phase 4 turns into token-level BIO tags.

Hugging Face concepts used here
-------------------------------
* ``Dataset`` — a table backed by Apache Arrow (columnar, memory-mapped from disk), so it can be
  larger than RAM and is fast to slice, ``.map()`` and ``.filter()``.
* ``Features`` — the dataset's schema. It fixes the type of each column, so every later phase
  knows exactly what it gets back.
* ``ClassLabel`` — a feature type that stores labels as integers (0..N-1) but remembers their
  names. ``ds.features["label"].int2str(3)`` -> ``"threat"``. This is the same mapping as
  ``label2id`` / ``id2label`` in config.py, which the Phase 3 model will need.
* ``DatasetDict`` — a dict of named splits (``train`` / ``validation`` / ``test``).
  ``save_to_disk`` writes it as Arrow files, and ``load_from_disk`` loads it back in one call.

Labeling rules that make some examples deliberately "hard"
-----------------------------------------------------------
* A successful login (IOS ``LOGIN_SUCCESS``, ISE ``5200``) is ``noise``, not ``auth_failure``.
  So are a successful 802.1X/MAB session (``SESSION_MGR-5-SUCCESS``), routine CoA (``5417``)
  and RADIUS accounting (``3000``/``3001``).
* A config command logged through TACACS+ accounting (ISE ``3300``) is ``config_change``.
* A PAN-OS TRAFFIC log is ``noise`` even when the action is ``deny`` (routine policy
  enforcement). Only THREAT logs are ``threat``.
* Any GlobalProtect log is ``vpn``, including GlobalProtect authentication failures.
* An interface up/down on the WLC (its uplinks) is ``link_event``, not ``wireless``.
* Routine messages from the WLC (NTP sync, a successful web-UI admin login) are ``noise``,
  not ``wireless``; a client excluded for a wrong PSK is ``auth_failure``.
* Authentication failures are ``auth_failure`` on every vendor, including WLC 802.1X
  failures (but GlobalProtect failures stay ``vpn``, see above).
* Security detections are ``threat`` on every vendor: PAN-OS THREAT, IOS IPS / ARP
  inspection, WLC rogue AP, ISE threat-centric NAC quarantine.

Anonymization: only RFC 5737 / RFC 1918 IPs, and fake names like ``user07`` or ``corp-sw-03``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from logsense.config import label_maps, load_config

# A template field is either a plain string, or an (value, entity_type) tuple for an entity.
Field = str | tuple[str, str]
Template = Callable[[random.Random, datetime], tuple[str, dict[str, Field]]]

# ---------------------------------------------------------------------------
# Fake-but-realistic field values
# ---------------------------------------------------------------------------

DOC_NETS = ["192.0.2", "198.51.100", "203.0.113"]  # RFC 5737: stand-ins for "internet" hosts
RULES = [
    "allow-web-out",
    "allow-dns",
    "block-tor",
    "deny-inbound-any",
    "allow-o365",
    "block-known-bad",
    "allow-vpn-users",
    "dmz-to-db",
    "guest-internet",
    "inbound-web-dmz",
]
SSIDS = ["CORP-WIFI", "CORP-GUEST", "CORP-IOT", "CORP-VOICE"]


def _private_ip(rng: random.Random) -> str:
    """Random RFC 1918 address."""
    kind = rng.randrange(3)
    if kind == 0:
        return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    if kind == 1:
        return f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _public_ip(rng: random.Random) -> str:
    """Random RFC 5737 documentation address (plays the role of a public IP)."""
    return f"{rng.choice(DOC_NETS)}.{rng.randint(1, 254)}"


def _user(rng: random.Random) -> str:
    return f"{rng.choice(['user', 'user', 'admin', 'svc'])}{rng.randint(1, 40):02d}"


def _host(rng: random.Random, kind: str) -> str:
    return f"corp-{kind}-{rng.randint(1, 12):02d}"


def _mac_octets(rng: random.Random) -> list[str]:
    return [f"{rng.randint(0, 255):02x}" for _ in range(6)]


def _mac_cisco(rng: random.Random) -> str:
    o = _mac_octets(rng)
    return f"{o[0]}{o[1]}.{o[2]}{o[3]}.{o[4]}{o[5]}"


def _mac_dash(rng: random.Random) -> str:
    return "-".join(_mac_octets(rng)).upper()


def _ios_iface(rng: random.Random) -> str:
    return rng.choice(
        [
            f"GigabitEthernet1/0/{rng.randint(1, 48)}",
            f"GigabitEthernet0/0/{rng.randint(0, 3)}",
            f"TenGigabitEthernet1/1/{rng.randint(1, 4)}",
            f"Port-channel{rng.randint(1, 8)}",
        ]
    )


def _pan_iface(rng: random.Random) -> str:
    return f"ethernet1/{rng.randint(1, 24)}"


def _vlan(rng: random.Random) -> str:
    return str(rng.randint(2, 4000))


def _e(value: str, etype: str) -> tuple[str, str]:
    """Mark a template value as an entity of type ``etype``."""
    return (value, etype)


# ---------------------------------------------------------------------------
# Vendor headers (the syslog prefix; hostnames are not entities)
# ---------------------------------------------------------------------------


def _syslog_ts(dt: datetime) -> str:
    return f"{dt:%b} {dt.day:>2} {dt:%H:%M:%S}"


def _ios_hdr(rng: random.Random, dt: datetime, kind: str = "sw") -> str:
    ms = rng.randint(0, 999)
    return f"{_syslog_ts(dt)} {_host(rng, kind)} {rng.randint(100, 999999)}: *{_syslog_ts(dt)}.{ms:03d} UTC: "


def _wlc_hdr(rng: random.Random, dt: datetime) -> str:
    return f"{_syslog_ts(dt)} {_host(rng, 'wlc')}: *{_syslog_ts(dt)}.{rng.randint(0, 999):03d}: "


def _ise_hdr(rng: random.Random, dt: datetime, category: str, code: int, sev: str) -> str:
    ts = f"{dt:%Y-%m-%d %H:%M:%S}.{rng.randint(0, 999):03d} +00:00"
    return (
        f"{_syslog_ts(dt)} {_host(rng, 'ise')} {category} {rng.randint(1, 10**9):010d} 1 0 "
        f"{ts} {rng.randint(1, 10**9):010d} {code} {sev} "
    )


def _pan_hdr(rng: random.Random, dt: datetime, log_type: str, subtype: str) -> str:
    """PAN-OS CSV prefix up to and including Generated Time (fields are comma-separated)."""
    ts = f"{dt:%Y/%m/%d %H:%M:%S}"
    serial = f"0128010{rng.randint(10000, 99999)}"
    return f"{_syslog_ts(dt)} {_host(rng, 'fw')} 1,{ts},{serial},{log_type},{subtype},2561,{ts},"


def _login_time(dt: datetime) -> str:
    return f"{dt:%H:%M:%S} UTC {dt:%a %b} {dt.day} {dt.year}"


# ---------------------------------------------------------------------------
# Templates — each returns (template_string, fields). "{name}" placeholders are filled from fields.
# ---------------------------------------------------------------------------

# --- auth_failure ---


def ios_login_failed(rng, dt):
    tpl = _ios_hdr(rng, dt) + (
        "%SEC_LOGIN-4-LOGIN_FAILED: Login failed [user: {user}] [Source: {src}] "
        "[localport: 22] [Reason: Login Authentication Failed] at {t}"
    )
    return tpl, {
        "user": _e(_user(rng), "USER"),
        "src": _e(_private_ip(rng), "IP"),
        "t": _login_time(dt),
    }


def ios_ssh_failed(rng, dt):
    tpl = _ios_hdr(rng, dt) + (
        "%SSH-5-SSH2_USERAUTH: User '{user}' authentication for SSH2 Session from {src} "
        "(tty = 0) using crypto cipher 'aes256-ctr', hmac 'hmac-sha2-256' Failed"
    )
    return tpl, {"user": _e(_user(rng), "USER"), "src": _e(_private_ip(rng), "IP")}


def ise_5400(rng, dt):
    reason = rng.choice(
        [
            (
                "24408 User authentication against Active Directory failed since user has "
                "entered the wrong password"
            ),
            "22056 Subject not found in the applicable identity store(s)",
            (
                "12514 EAP-TLS failed SSL/TLS handshake because of an unknown CA in the client "
                "certificates chain"
            ),
        ]
    )
    tpl = _ise_hdr(rng, dt, "CISE_Failed_Attempts", 5400, "NOTICE") + (
        "Failed-Attempt: Authentication failed, ConfigVersionId={cv}, Device IP Address={nas}, "
        "Device Port=1812, UserName={user}, Calling-Station-ID={mac}, NAS-Port-Id={iface}, "
        "FailureReason={reason}"
    )
    return tpl, {
        "cv": str(rng.randint(10, 400)),
        "nas": _e(_private_ip(rng), "IP"),
        "user": _e(_user(rng), "USER"),
        "mac": _e(_mac_dash(rng), "MAC"),
        "iface": _e(_ios_iface(rng), "INTERFACE"),
        "reason": reason,
    }


def ise_5405(rng, dt):
    tpl = _ise_hdr(rng, dt, "CISE_Failed_Attempts", 5405, "NOTICE") + (
        "RADIUS: RADIUS Request dropped, ConfigVersionId={cv}, Device IP Address={nas}, "
        "UserName={user}, Calling-Station-ID={mac}, FailureReason={reason}"
    )
    return tpl, {
        # 11007 = the switch isn't defined as a network device in ISE ("misconfigured NAD")
        "reason": rng.choice(
            [
                "11036 The Message-Authenticator RADIUS attribute is invalid",
                "11007 Could not locate Network Device or AAA Client",
            ]
        ),
        "cv": str(rng.randint(10, 400)),
        "nas": _e(_private_ip(rng), "IP"),
        "user": _e(_user(rng), "USER"),
        "mac": _e(_mac_dash(rng), "MAC"),
    }


def pan_auth_fail(rng, dt):
    tpl = _pan_hdr(rng, dt, "SYSTEM", "auth") + (
        ",auth-fail,,0,0,general,medium,\"failed authentication for user '{user}'. "
        'Reason: Invalid username/password. From: {src}.",{seq},0x0,0,0,0,0,,'
    )
    return tpl, {
        "user": _e(_user(rng), "USER"),
        "src": _e(_private_ip(rng), "IP"),
        "seq": str(rng.randint(10**6, 10**7)),
    }


# --- config_change ---


def ios_config_i(rng, dt, kind="sw"):
    hdr = _wlc_hdr(rng, dt) if kind == "wlc" else _ios_hdr(rng, dt, kind)
    tpl = hdr + "%SYS-5-CONFIG_I: Configured from console by {user} on vty{n} ({src})"
    return tpl, {
        "user": _e(_user(rng), "USER"),
        "n": str(rng.randint(0, 4)),
        "src": _e(_private_ip(rng), "IP"),
    }


def ios_parser_cmd(rng, dt):
    tpl = _ios_hdr(rng, dt) + "%PARSER-5-CFGLOG_LOGGEDCMD: User:{user}  logged command:"
    fields: dict[str, Field] = {"user": _e(_user(rng), "USER")}
    if rng.random() < 0.5:
        tpl += "interface {iface}"
        fields["iface"] = _e(_ios_iface(rng), "INTERFACE")
    else:
        tpl += "switchport access vlan {vlan}"
        fields["vlan"] = _e(_vlan(rng), "VLAN")
    return tpl, fields


def ise_config(rng, dt):
    op = rng.choice(["Added", "Changed", "Deleted"])
    obj = rng.choice(["Network Device", "Authorization Profile", "Internal User"])
    tpl = _ise_hdr(rng, dt, "CISE_Administrative_and_Operational_Audit", 52000, "NOTICE") + (
        "Configuration-Changes: {op} configuration, ConfigVersionId={cv}, AdminInterface=GUI, "
        "AdminIPAddress={src}, AdminName={user}, OperationMessageText={op2} {obj}, "
        "ObjectType={obj2}"
    )
    return tpl, {
        "op": op,
        "op2": op,
        "obj": obj,
        "obj2": obj,
        "cv": str(rng.randint(10, 400)),
        "src": _e(_private_ip(rng), "IP"),
        "user": _e(_user(rng), "USER"),
    }


def pan_config(rng, dt):
    tpl = _pan_hdr(rng, dt, "CONFIG", "0") + (
        '{src},vsys1,{cmd},{user},Web,Submitted," vsys  vsys1 rulebase security rules  {rule}",'
        "{seq},0x0,0,0,0,0,,"
    )
    return tpl, {
        "src": _e(_private_ip(rng), "IP"),
        "cmd": rng.choice(["set", "edit", "delete"]),
        "user": _e(_user(rng), "USER"),
        "rule": _e(rng.choice(RULES), "RULE"),
        "seq": str(rng.randint(10**6, 10**7)),
    }


# --- link_event ---


def ios_link_updown(rng, dt, kind="sw"):
    hdr = _wlc_hdr(rng, dt) if kind == "wlc" else _ios_hdr(rng, dt, kind)
    tpl = hdr + "%LINK-3-UPDOWN: Interface {iface}, changed state to {st}"
    return tpl, {"iface": _e(_ios_iface(rng), "INTERFACE"), "st": rng.choice(["up", "down"])}


def ios_lineproto(rng, dt):
    tpl = _ios_hdr(rng, dt, rng.choice(["sw", "rtr"])) + (
        "%LINEPROTO-5-UPDOWN: Line protocol on Interface {iface}, changed state to {st}"
    )
    return tpl, {"iface": _e(_ios_iface(rng), "INTERFACE"), "st": rng.choice(["up", "down"])}


def ios_admin_down(rng, dt):
    tpl = _ios_hdr(rng, dt) + (
        "%LINK-5-CHANGED: Interface {iface}, changed state to administratively down"
    )
    return tpl, {"iface": _e(_ios_iface(rng), "INTERFACE")}


def pan_link_change(rng, dt):
    iface = _pan_iface(rng)
    up = rng.random() < 0.5
    tpl = _pan_hdr(rng, dt, "SYSTEM", "general") + (
        ',link-change,{iface},0,0,general,{sev},"Port {iface2}: {state} 1Gb/s-full duplex",'
        "{seq},0x0,0,0,0,0,,"
    )
    return tpl, {
        "iface": _e(iface, "INTERFACE"),
        "iface2": _e(iface, "INTERFACE"),
        "sev": "informational" if up else "critical",
        "state": "Up" if up else "Down",
        "seq": str(rng.randint(10**6, 10**7)),
    }


# --- threat ---

THREATS = {
    "vulnerability": [
        ("Apache Log4j Remote Code Execution Vulnerability", 91991, "critical"),
        ("Microsoft Windows SMB Remote Code Execution Vulnerability", 33181, "critical"),
        ("SSH User Authentication Brute Force Attempt", 40015, "high"),
        ("HTTP Directory Traversal Vulnerability", 30844, "medium"),
    ],
    "spyware": [
        ("Cobalt Strike Beacon Command and Control Traffic Detection", 86246, "critical"),
        ("Suspicious DNS Query (generic:malware-c2.example)", 109001, "medium"),
    ],
    "virus": [
        ("Virus/Win32.WGeneric.dnvmke", 274051, "medium"),
        ("Trojan/Win32.emotet.bqrt", 229003, "high"),
    ],
}


def pan_threat(rng, dt):
    subtype = rng.choice(list(THREATS))
    name, tid, sev = rng.choice(THREATS[subtype])
    inbound = subtype == "vulnerability"
    src, dst = (
        (_public_ip(rng), _private_ip(rng)) if inbound else (_private_ip(rng), _public_ip(rng))
    )
    src_user = _user(rng) if not inbound and rng.random() < 0.6 else ""
    tpl = _pan_hdr(rng, dt, "THREAT", subtype) + (
        "{src},{dst},,,{rule},{suser},,{app},vsys1,{fz},{tz},{inif},{outif},Log-Fwd,{sid},1,"
        '{sport},{dport},0,0,0x2000,tcp,{action},"",{name}({tid}),any,{sev},client-to-server'
    )
    return tpl, {
        "src": _e(src, "IP"),
        "dst": _e(dst, "IP"),
        "rule": _e(rng.choice(RULES), "RULE"),
        "suser": _e(src_user, "USER"),
        "app": rng.choice(["web-browsing", "ssl", "smb", "dns"]),
        "fz": "untrust" if inbound else "trust",
        "tz": "trust" if inbound else "untrust",
        "inif": _e(_pan_iface(rng), "INTERFACE"),
        "outif": _e(_pan_iface(rng), "INTERFACE"),
        "sid": str(rng.randint(10**4, 10**6)),
        "sport": str(rng.randint(1024, 65535)),
        "dport": rng.choice(["80", "443", "445", "53", "22"]),
        "action": rng.choice(["reset-both", "drop", "alert", "block-ip"]),
        "name": name,
        "tid": str(tid),
        "sev": sev,
    }


def ios_ips_signature(rng, dt):
    sig, name = rng.choice(
        [
            (3051, "TCP Connection Window Size DoS"),
            (5477, "SQL Injection Attempt"),
            (1102, "Impossible IP Packet"),
            (3041, "TCP SYN/FIN Packet"),
        ]
    )
    tpl = _ios_hdr(rng, dt, "rtr") + (
        "%IPS-4-SIGNATURE: Sig:{sig} Subsig:0 Sev:{sev} {name} [{src}:{sport} -> {dst}:{dport}]"
    )
    return tpl, {
        "sig": str(sig),
        "sev": str(rng.choice([25, 50, 75, 100])),
        "name": name,
        "src": _e(_public_ip(rng), "IP"),
        "sport": str(rng.randint(1024, 65535)),
        "dst": _e(_private_ip(rng), "IP"),
        "dport": rng.choice(["80", "443", "22"]),
    }


# --- vpn ---


def pan_globalprotect(rng, dt):
    event, stage, status, desc = rng.choice(
        [
            (
                "gateway-connected",
                "connected",
                "success",
                "GlobalProtect gateway client connected.",
            ),
            ("gateway-logout", "logout", "success", "GlobalProtect gateway user logout succeeded."),
            (
                "portal-auth",
                "login",
                "failure",
                "Authentication failed: Invalid username or password",
            ),
            (
                "gateway-auth",
                "login",
                "success",
                "GlobalProtect gateway user authentication succeeded.",
            ),
        ]
    )
    tpl = _pan_hdr(rng, dt, "GLOBALPROTECT", "0") + (
        "vsys1,{event},{stage},LDAP,IPSec,{user},US,corp-lt-{nn},{pub},,{priv},,,,6.2.1,"
        'Microsoft Windows,10.0,1,,,"{desc}",{status},,0,on-demand'
    )
    return tpl, {
        "event": event,
        "stage": stage,
        "user": _e(_user(rng), "USER"),
        "nn": f"{rng.randint(1, 99):02d}",
        "pub": _e(_public_ip(rng), "IP"),
        "priv": _e(_private_ip(rng), "IP"),
        "desc": desc,
        "status": status,
    }


def ios_crypto_session(rng, dt):
    peer = _public_ip(rng)
    tpl = _ios_hdr(rng, dt, "rtr") + (
        "%CRYPTO-5-SESSION_STATUS: Crypto tunnel is {st}. Peer {peer}:500 Id: {peer2}"
    )
    return tpl, {"st": rng.choice(["UP", "DOWN"]), "peer": _e(peer, "IP"), "peer2": _e(peer, "IP")}


def ios_crypto_inv_spi(rng, dt):
    tpl = _ios_hdr(rng, dt, "rtr") + (
        "%CRYPTO-4-RECVD_PKT_INV_SPI: decaps: rec'd IPSEC packet has invalid spi for "
        "destaddr={dst}, prot=50, spi=0x{spi}, srcaddr={src}, input interface {iface}"
    )
    return tpl, {
        "dst": _e(_public_ip(rng), "IP"),
        "spi": f"{rng.randint(0, 2**32 - 1):08X}",
        "src": _e(_public_ip(rng), "IP"),
        "iface": _e(_ios_iface(rng), "INTERFACE"),
    }


def pan_ike(rng, dt):
    event, sev, msg = rng.choice(
        [
            (
                "ike-nego-p1-fail",
                "high",
                "IKE phase-1 negotiation is failed as initiator, main mode.",
            ),
            ("ike-nego-p2-succ", "informational", "IKE phase-2 negotiation is succeeded."),
            ("tunnel-status", "informational", "Tunnel is up."),
        ]
    )
    tpl = _pan_hdr(rng, dt, "SYSTEM", "vpn") + (
        ',{event},tunnel-to-branch{n},0,0,ike,{sev},"{msg} SA: {local}[500]-{peer}[500].",'
        "{seq},0x0,0,0,0,0,,"
    )
    return tpl, {
        "event": event,
        "n": str(rng.randint(1, 9)),
        "sev": sev,
        "msg": msg,
        "local": _e(_public_ip(rng), "IP"),
        "peer": _e(_public_ip(rng), "IP"),
        "seq": str(rng.randint(10**6, 10**7)),
    }


# --- wireless ---


def wlc_client_run(rng, dt):
    tpl = _wlc_hdr(rng, dt) + (
        "%CLIENT_ORCH_LOG-6-CLIENT_ADDED_TO_RUN_STATE: Chassis 1 R0/0: wncd: Username entry "
        "({user}) joined with ssid ({ssid}) for device with MAC: {mac} on channel ({ch})"
    )
    return tpl, {
        "user": _e(_user(rng), "USER"),
        "ssid": rng.choice(SSIDS),
        "mac": _e(_mac_cisco(rng), "MAC"),
        "ch": str(rng.choice([1, 6, 11, 36, 44, 149])),
    }


def wlc_client_deauth(rng, dt):
    reason = rng.choice(
        [
            "DEAUTH_REASON_INACTIVITY_TIMEOUT",
            "CO_CLIENT_DELETE_REASON_MN_AP_DISASSOC",
            "CLIENT_DELETE_REASON_SESSION_TIMEOUT",
            "CO_CLIENT_DELETE_REASON_DOT11_DEAUTH",
            # verbatim reason names from BRKEWN-3628
            "CO_CLIENT_DELETE_REASON_DEAUTH_OR_DISASSOC_REQ",
            "CO_CLIENT_DELETE_REASON_GROUP_KEY_UPDATE_TIMEOUT",
        ]
    )
    tpl = _wlc_hdr(rng, dt) + (
        "%CLIENT_ORCH_LOG-5-CLIENT_DEAUTH: Chassis 1 R0/0: wncd: Client {mac} on AP "
        "corp-ap-{nn} slot {slot} deauthenticated, reason: {reason}, vlan {vlan}"
    )
    return tpl, {
        "mac": _e(_mac_cisco(rng), "MAC"),
        "nn": f"{rng.randint(1, 60):02d}",
        "slot": str(rng.randint(0, 1)),
        "reason": reason,
        "vlan": _e(_vlan(rng), "VLAN"),
    }


def wlc_ap_join(rng, dt):
    tpl = _wlc_hdr(rng, dt) + (
        "%CAPWAPAC_SMGR_TRACE_MESSAGE-5-AP_JOIN_DISJOIN: Chassis 1 R0/0: wncd: AP Event: "
        "AP Name: corp-ap-{nn}, MAC: {mac} {ev}"
    )
    return tpl, {
        "nn": f"{rng.randint(1, 60):02d}",
        "mac": _e(_mac_cisco(rng), "MAC"),
        "ev": rng.choice(["Joined", "Disjoined"]),
    }


def wlc_apmgr(rng, dt):
    msg = rng.choice(
        [
            "Radar detected on channel 52, changing channel on slot 1",
            "Radio slot 1 went down due to admin state change",
            "AP rebooted, reason: image upgrade",
        ]
    )
    tpl = _wlc_hdr(rng, dt) + (
        "%APMGR_TRACE_MESSAGE-3-AP_EVENT: Chassis 1 R0/0: wncd: AP corp-ap-{nn} ({mac}): {msg}"
    )
    return tpl, {"nn": f"{rng.randint(1, 60):02d}", "mac": _e(_mac_cisco(rng), "MAC"), "msg": msg}


# --- noise ---


def ntp_sync(rng, dt, kind="sw"):
    hdr = _wlc_hdr(rng, dt) if kind == "wlc" else _ios_hdr(rng, dt, kind)
    msg = rng.choice(
        ["%NTP-5-PEERSYNC: NTP synced to peer {ip}", "%NTP-6-PEERREACH: Peer {ip} is reachable"]
    )
    return hdr + msg, {"ip": _e(_private_ip(rng), "IP")}


def cdp_native_vlan(rng, dt):
    tpl = _ios_hdr(rng, dt) + (
        "%CDP-4-NATIVE_VLAN_MISMATCH: Native VLAN mismatch discovered on {if1} ({v1}), "
        "with corp-sw-{nn} {if2} ({v2})."
    )
    return tpl, {
        "if1": _e(_ios_iface(rng), "INTERFACE"),
        "v1": _e(_vlan(rng), "VLAN"),
        "nn": f"{rng.randint(1, 12):02d}",
        "if2": _e(_ios_iface(rng), "INTERFACE"),
        "v2": _e(_vlan(rng), "VLAN"),
    }


def ios_login_success(rng, dt):
    tpl = _ios_hdr(rng, dt) + (
        "%SEC_LOGIN-5-LOGIN_SUCCESS: Login Success [user: {user}] [Source: {src}] "
        "[localport: 22] at {t}"
    )
    return tpl, {
        "user": _e(_user(rng), "USER"),
        "src": _e(_private_ip(rng), "IP"),
        "t": _login_time(dt),
    }


def ise_5200(rng, dt):
    tpl = _ise_hdr(rng, dt, "CISE_Passed_Authentications", 5200, "NOTICE") + (
        "Passed-Authentication: Authentication succeeded, ConfigVersionId={cv}, "
        "Device IP Address={nas}, Device Port=1812, UserName={user}, Calling-Station-ID={mac}, "
        "NAS-Port-Id={iface}, AuthorizationPolicyMatchedRule=Corp-Wired-Dot1x"
    )
    return tpl, {
        "cv": str(rng.randint(10, 400)),
        "nas": _e(_private_ip(rng), "IP"),
        "user": _e(_user(rng), "USER"),
        "mac": _e(_mac_dash(rng), "MAC"),
        "iface": _e(_ios_iface(rng), "INTERFACE"),
    }


def pan_traffic(rng, dt):
    action = rng.choice(["allow", "allow", "deny", "drop"])
    subtype = "end" if action == "allow" else action
    tpl = _pan_hdr(rng, dt, "TRAFFIC", subtype) + (
        "{src},{dst},,,{rule},,,{app},vsys1,trust,untrust,{inif},{outif},Log-Fwd,{sid},1,"
        "{sport},{dport},0,0,0x400064,tcp,{action},{bytes},{sent},{recv},{pkts},"
        "{start},{elapsed},any,0,{seq},0x0,any,United States,0,{pkts},0"
    )
    sent, recv = rng.randint(60, 50000), rng.randint(0, 500000)
    return tpl, {
        "src": _e(_private_ip(rng), "IP"),
        "dst": _e(_public_ip(rng), "IP"),
        "rule": _e(rng.choice(RULES), "RULE"),
        "app": rng.choice(["web-browsing", "ssl", "dns", "ntp", "office365-base"]),
        "inif": _e(_pan_iface(rng), "INTERFACE"),
        "outif": _e(_pan_iface(rng), "INTERFACE"),
        "sid": str(rng.randint(10**4, 10**6)),
        "sport": str(rng.randint(1024, 65535)),
        "dport": rng.choice(["80", "443", "53", "123"]),
        "action": action,
        "bytes": str(sent + recv),
        "sent": str(sent),
        "recv": str(recv),
        "pkts": str(rng.randint(1, 400)),
        "start": f"{dt:%Y/%m/%d %H:%M:%S}",
        "elapsed": str(rng.randint(0, 600)),
        "seq": str(rng.randint(10**6, 10**7)),
    }


def ios_logging_host(rng, dt):
    tpl = _ios_hdr(rng, dt) + (
        "%SYS-6-LOGGINGHOST_STARTSTOP: Logging to host {ip} port 514 started - CLI initiated"
    )
    return tpl, {"ip": _e(_private_ip(rng), "IP")}


def pan_scheduled_job(rng, dt):
    job = rng.choice(
        [
            "Content update job",
            "Antivirus update job",
            "WildFire update job",
            "Scheduled config backup",
        ]
    )
    tpl = _pan_hdr(rng, dt, "SYSTEM", "general") + (
        ',general,,0,0,general,informational,"{job} completed successfully",{seq},0x0,0,0,0,0,,'
    )
    return tpl, {"job": job, "seq": str(rng.randint(10**6, 10**7))}


# --- extra templates so each label spans several vendor formats (breaks the
#     "vendor format -> label" shortcut the Phase 3 model learned) ---


def wlc_dot1x_fail(rng, dt):  # auth_failure on the WLC
    reason = rng.choice(["Cred Fail", "Timeout", "No Response from Client", "AAA Server Down"])
    tpl = _wlc_hdr(rng, dt) + (
        "%DOT1X-5-FAIL: Chassis 1 R0/0: wncd: Authentication failed for client ({mac}) with "
        "reason ({reason}) on Interface capwap_{cap} AuditSessionID {sess} Username: {user}"
    )
    return tpl, {
        "mac": _e(_mac_cisco(rng), "MAC"),
        "reason": reason,
        "cap": f"9000{rng.randint(0, 9999):04d}",
        "sess": f"{rng.randint(0, 16**24 - 1):024X}",
        "user": _e(_user(rng), "USER"),
    }


def pan_auth_log(rng, dt):  # auth_failure, PAN-OS AUTH log type
    user = _user(rng)
    tpl = _pan_hdr(rng, dt, "AUTH", "0") + (
        "vsys1,{src},{user},corp\\{user2},{obj},auth-policy-{n},1,0,,Log-Fwd,ldap-corp,"
        '"{desc}",0,auth-fail,1'
    )
    return tpl, {
        "src": _e(_private_ip(rng), "IP"),
        "user": _e(user, "USER"),
        "user2": _e(user, "USER"),
        "obj": rng.choice(["captive-portal", "mfa-portal", "admin-ui"]),
        "n": str(rng.randint(1, 5)),
        "desc": rng.choice(["Invalid credentials", "Account locked", "MFA push denied"]),
    }


def ise_policy_change(rng, dt):  # config_change, a second ISE format
    obj = rng.choice(["Corp-Wired-Dot1x", "Corp-Wireless-PEAP", "Guest-Redirect"])
    tpl = _ise_hdr(rng, dt, "CISE_Administrative_and_Operational_Audit", 51001, "INFO") + (
        "Policy-Changes: Policy rule modified, AdminInterface=CLI, AdminIPAddress={src}, "
        "AdminName={user}, ObjectName={obj}, ObjectType=Authorization Rule, Status=Success"
    )
    return tpl, {"src": _e(_private_ip(rng), "IP"), "user": _e(_user(rng), "USER"), "obj": obj}


def pan_ha_link(rng, dt):  # link_event, a second PAN-OS format
    link, state = rng.choice(["HA1", "HA2", "HA1-backup"]), rng.choice(["up", "down"])
    tpl = _pan_hdr(rng, dt, "SYSTEM", "ha") + (
        ',ha-link-change,{iface},0,0,ha,{sev},"{link} link {state} on {iface2}",{seq},0x0,0,0,0,0,,'
    )
    iface = _pan_iface(rng)
    return tpl, {
        "iface": _e(iface, "INTERFACE"),
        "iface2": _e(iface, "INTERFACE"),
        "sev": "critical" if state == "down" else "informational",
        "link": link,
        "state": state,
        "seq": str(rng.randint(10**6, 10**7)),
    }


def ios_dai_deny(rng, dt):  # threat on IOS: ARP spoofing caught by Dynamic ARP Inspection
    tpl = _ios_hdr(rng, dt) + (
        "%SW_DAI-4-DHCP_SNOOPING_DENY: 1 Invalid ARPs (Req) on {iface}, vlan {vlan}."
        "([{mac}/{ip}/0000.0000.0000/{ip2}/{t}])"
    )
    return tpl, {
        "iface": _e(_ios_iface(rng), "INTERFACE"),
        "vlan": _e(_vlan(rng), "VLAN"),
        "mac": _e(_mac_cisco(rng), "MAC"),
        "ip": _e(_private_ip(rng), "IP"),
        "ip2": _e(_private_ip(rng), "IP"),
        "t": _login_time(dt),
    }


def wlc_rogue_ap(rng, dt):  # threat on the WLC
    tpl = _wlc_hdr(rng, dt) + (
        "%ROGUE-4-AP_DETECTED: Chassis 1 R0/0: wncd: Malicious rogue AP {mac} detected on "
        "channel {ch} by AP corp-ap-{nn}, impersonating SSID ({ssid}), RSSI {rssi} dBm"
    )
    return tpl, {
        "mac": _e(_mac_cisco(rng), "MAC"),
        "ch": str(rng.choice([1, 6, 11, 36, 149])),
        "nn": f"{rng.randint(1, 60):02d}",
        "ssid": rng.choice(SSIDS),
        "rssi": str(-rng.randint(40, 85)),
    }


def ise_anc_quarantine(rng, dt):  # threat on ISE: threat-centric NAC quarantine
    tpl = _ise_hdr(rng, dt, "CISE_Threat_Centric_NAC", 91002, "WARN") + (
        "TC-NAC: Endpoint quarantined, EndpointMacAddress={mac}, IPAddress={ip}, "
        "UserName={user}, ThreatSeverity=High, AdapterName=AMP, Indicator={ind}"
    )
    return tpl, {
        "mac": _e(_mac_dash(rng), "MAC"),
        "ip": _e(_private_ip(rng), "IP"),
        "user": _e(_user(rng), "USER"),
        "ind": rng.choice(["Ransomware.Generic", "Trojan.Emotet", "Backdoor.CobaltStrike"]),
    }


def ise_profiler(rng, dt):  # noise on ISE: routine endpoint profiling
    tpl = _ise_hdr(rng, dt, "CISE_Profiler", 80002, "INFO") + (
        "Profiler: Profiler EndPoint profiling event occurred, EndpointMacAddress={mac}, "
        "EndpointPolicy={pol}, EndpointIPAddress={ip}, NADAddress={nas}"
    )
    return tpl, {
        "mac": _e(_mac_dash(rng), "MAC"),
        "pol": rng.choice(["Apple-iPhone", "Windows10-Workstation", "Cisco-IP-Phone", "Printer"]),
        "ip": _e(_private_ip(rng), "IP"),
        "nas": _e(_private_ip(rng), "IP"),
    }


# --- data v2: ISE / IBNS 2.0 events, selected from Cisco Live 2026 BRKSEC-2347 (ISE deployment
#     tips: 802.1X/MAB policies, live-log failures, Easy Connect CoA, TACACS+) and BRKSEC-2660
#     (RADIUS accounting). Formats from public Cisco docs; values are fake. ---


def _ios_iface_short(rng: random.Random) -> str:
    return rng.choice([f"Gi1/0/{rng.randint(1, 48)}", f"Gi2/0/{rng.randint(1, 48)}"])


def _session_hdr(rng: random.Random, dt: datetime, mnemonic: str) -> str:
    """IOS-XE session-manager prefix, e.g. ``%DOT1X-5-FAIL: Switch 1 R0/0: sessmgrd: ``."""
    return _ios_hdr(rng, dt) + f"%{mnemonic}: Switch {rng.randint(1, 2)} R0/0: sessmgrd: "


def _audit_id(rng: random.Random) -> str:
    return f"{rng.randint(0, 16**24 - 1):024X}"


def ios_dot1x_mab_fail(rng, dt):  # auth_failure: 802.1X or MAB failed on a switchport
    if rng.random() < 0.5:
        tpl = _session_hdr(rng, dt, "DOT1X-5-FAIL") + (
            "Authentication failed for client ({mac}) with reason ({reason}) on Interface "
            "{iface} AuditSessionID {sess} Username: {user}"
        )
        fields: dict[str, Field] = {
            "reason": rng.choice(["Cred Fail", "Timeout", "No Response from Client"]),
            "user": _e(_user(rng), "USER"),
        }
    else:
        tpl = _session_hdr(rng, dt, "MAB-5-FAIL") + (
            "Authentication failed for client ({mac}) on Interface {iface} AuditSessionID {sess}"
        )
        fields = {}
    return tpl, {
        **fields,
        "mac": _e(_mac_cisco(rng), "MAC"),
        "iface": _e(_ios_iface_short(rng), "INTERFACE"),
        "sess": _audit_id(rng),
    }


def ios_session_fail(rng, dt):  # auth_failure: authorization failed / not applied
    tpl = _session_hdr(rng, dt, "SESSION_MGR-5-FAIL") + (
        "Authorization failed or unapplied for client ({mac}) method ({method}) on Interface "
        "{iface} AuditSessionID {sess}"
    )
    return tpl, {
        "mac": _e(_mac_cisco(rng), "MAC"),
        "method": rng.choice(["dot1x", "mab"]),
        "iface": _e(_ios_iface_short(rng), "INTERFACE"),
        "sess": _audit_id(rng),
    }


def ios_session_success(rng, dt):  # noise (hard example): a normal successful session
    tpl = _session_hdr(rng, dt, "SESSION_MGR-5-SUCCESS") + (
        "Authorization succeeded for client ({mac}) on Interface {iface} AuditSessionID {sess}"
    )
    return tpl, {
        "mac": _e(_mac_cisco(rng), "MAC"),
        "iface": _e(_ios_iface_short(rng), "INTERFACE"),
        "sess": _audit_id(rng),
    }


def _ise_radius_fields(rng: random.Random) -> dict[str, Field]:
    return {
        "cv": str(rng.randint(10, 400)),
        "nas": _e(_private_ip(rng), "IP"),
        "user": _e(_user(rng), "USER"),
        "mac": _e(_mac_dash(rng), "MAC"),
        "iface": _e(_ios_iface(rng), "INTERFACE"),
    }


def ise_5411(rng, dt):  # auth_failure: supplicant gave up mid-EAP (cert/supplicant issues)
    tpl = _ise_hdr(rng, dt, "CISE_Failed_Attempts", 5411, "NOTICE") + (
        "RADIUS: Supplicant stopped responding to ISE, ConfigVersionId={cv}, "
        "Device IP Address={nas}, UserName={user}, Calling-Station-ID={mac}, NAS-Port-Id={iface}, "
        "FailureReason=12321 PEAP failed SSL/TLS handshake because the client rejected the ISE "
        "local-certificate"
    )
    return tpl, _ise_radius_fields(rng)


def ise_5434(rng, dt):  # auth_failure: ISE collapsing repeated failures from one endpoint
    tpl = _ise_hdr(rng, dt, "CISE_Failed_Attempts", 5434, "NOTICE") + (
        "RADIUS: Endpoint conducted several failed authentications of the same scenario, "
        "ConfigVersionId={cv}, Device IP Address={nas}, UserName={user}, "
        "Calling-Station-ID={mac}, NAS-Port-Id={iface}, FailureReason=24408 User authentication "
        "against Active Directory failed since user has entered the wrong password"
    )
    return tpl, _ise_radius_fields(rng)


def ise_coa(rng, dt):  # noise: routine CoA (e.g. after profiling or an Easy Connect AD login)
    tpl = _ise_hdr(rng, dt, "CISE_Passed_Authentications", 5417, "NOTICE") + (
        "Dynamic-Authorization: Dynamic Authorization succeeded, ConfigVersionId={cv}, "
        "Device IP Address={nas}, UserName={user}, Calling-Station-ID={mac}, NAS-Port-Id={iface}, "
        "Event-Trigger={trigger}"
    )
    trigger = rng.choice(["Profiler-CoA", "EasyConnect-AD-Login", "Session-Reauthenticate"])
    return tpl, {**_ise_radius_fields(rng), "trigger": trigger}


def ise_accounting(rng, dt):  # noise: RADIUS accounting start / stop
    start = rng.random() < 0.5
    code, kind = (3000, "start") if start else (3001, "stop")
    tpl = _ise_hdr(rng, dt, "CISE_RADIUS_Accounting", code, "NOTICE") + (
        "Radius-Accounting: RADIUS Accounting {kind} request, ConfigVersionId={cv}, "
        "Device IP Address={nas}, UserName={user}, Calling-Station-ID={mac}, NAS-Port-Id={iface}, "
        "Acct-Status-Type={status}, Acct-Session-Id={acct}"
    )
    return tpl, {
        **_ise_radius_fields(rng),
        "kind": kind,
        "status": "Start" if start else "Stop",
        "acct": f"{rng.randint(0, 16**8 - 1):08X}",
    }


def ise_tacacs_cmd(rng, dt):  # config_change: a config command logged via TACACS+ accounting
    tpl = _ise_hdr(rng, dt, "CISE_TACACS_Accounting", 3300, "NOTICE") + (
        "TACACS-Accounting: TACACS+ Accounting with Command, ConfigVersionId={cv}, "
        "Device IP Address={nas}, UserName={user}, Remote-Address={src}, CmdSet=[ CmdAV="
    )
    fields: dict[str, Field] = {
        "cv": str(rng.randint(10, 400)),
        "nas": _e(_private_ip(rng), "IP"),
        "user": _e(_user(rng), "USER"),
        "src": _e(_private_ip(rng), "IP"),
    }
    if rng.random() < 0.5:
        tpl += "interface CmdArgAV={iface} CmdArgAV=<cr> ]"
        fields["iface"] = _e(_ios_iface(rng), "INTERFACE")
    else:
        tpl += "switchport CmdArgAV=access CmdArgAV=vlan CmdArgAV={vlan} CmdArgAV=<cr> ]"
        fields["vlan"] = _e(_vlan(rng), "VLAN")
    return tpl, fields


# --- data v2: Catalyst 9800 syslog formats shown verbatim in Cisco Live 2026 BRKEWN-3628
#     (Troubleshoot Catalyst 9800 Wireless Controllers). Values are fake. ---


def wlc_ap_heartbeat(rng, dt):  # wireless: AP lost its CAPWAP session to the controller
    tpl = _wlc_hdr(rng, dt) + (
        "%CAPWAPAC_SMGR_TRACE_MESSAGE-3-EWLC_GEN_ERR: Chassis 1 R0/0: wncd: Error in "
        "Session-IP: {ip}[{port}] Mac: {mac} Heartbeat timer expiry for AP. Close CAPWAP DTLS "
        "session"
    )
    return tpl, {
        "ip": _e(_private_ip(rng), "IP"),
        "port": str(rng.randint(5000, 65000)),
        "mac": _e(_mac_cisco(rng), "MAC"),
    }


def wlc_client_excluded(rng, dt):  # auth_failure (hard: a WLC line, but it's a bad PSK)
    tpl = _wlc_hdr(rng, dt) + (
        "%CLIENT_EXCLUSION_SERVER-5-ADD_TO_BLACKLIST_REASON: Chassis 1 R0/0: wncmgrd: "
        "Client MAC: {mac} was added to exclusion list, reason: Wrong PSK"
    )
    return tpl, {"mac": _e(_mac_cisco(rng), "MAC")}


def wlc_session_fail(rng, dt):  # auth_failure: wireless authorization failed
    tpl = _wlc_hdr(rng, dt) + (
        "%SESSION_MGR-5-FAIL: Chassis 1 R0/0: sessmgrd: Authorization failed or unapplied for "
        "client ({mac}) on Interface capwap_9000{cap} AuditSessionID {sess}"
    )
    return tpl, {
        "mac": _e(_mac_cisco(rng), "MAC"),
        "cap": f"{rng.randint(0, 0xFFFF):04x}",
        "sess": _audit_id(rng),
    }


def wlc_webui_login(rng, dt):  # noise: successful admin login to the 9800 web UI
    tpl = _wlc_hdr(rng, dt) + (
        "%WEBSERVER-5-LOGIN_PASSED: Chassis 1 R0/0: nginx: Login Successful from host {ip} by "
        "user '{user}' using crypto cipher 'ECDHE-RSA-AES128-GCM-SHA256'"
    )
    return tpl, {"ip": _e(_private_ip(rng), "IP"), "user": _e(_user(rng), "USER")}


# (vendor, template) pairs per label. Vendors are filtered by config `data.vendors`.
TEMPLATES: dict[str, list[tuple[str, Template]]] = {
    "auth_failure": [
        ("cisco_ios", ios_login_failed),
        ("cisco_ios", ios_ssh_failed),
        ("cisco_ise", ise_5400),
        ("cisco_ise", ise_5405),
        ("panos", pan_auth_fail),
        ("panos", pan_auth_log),
        ("cisco_wlc", wlc_dot1x_fail),
        ("cisco_ios", ios_dot1x_mab_fail),
        ("cisco_ios", ios_session_fail),
        ("cisco_ise", ise_5411),
        ("cisco_ise", ise_5434),
        ("cisco_wlc", wlc_client_excluded),
        ("cisco_wlc", wlc_session_fail),
    ],
    "config_change": [
        ("cisco_ios", ios_config_i),
        ("cisco_ios", ios_parser_cmd),
        ("cisco_wlc", partial(ios_config_i, kind="wlc")),
        ("cisco_ise", ise_config),
        ("cisco_ise", ise_policy_change),
        ("cisco_ise", ise_tacacs_cmd),
        ("panos", pan_config),
    ],
    "link_event": [
        ("cisco_ios", ios_link_updown),
        ("cisco_ios", ios_lineproto),
        ("cisco_ios", ios_admin_down),
        ("cisco_wlc", partial(ios_link_updown, kind="wlc")),
        ("panos", pan_link_change),
        ("panos", pan_ha_link),
    ],
    "threat": [
        ("panos", pan_threat),
        ("cisco_ios", ios_ips_signature),
        ("cisco_ios", ios_dai_deny),
        ("cisco_wlc", wlc_rogue_ap),
        ("cisco_ise", ise_anc_quarantine),
    ],
    "vpn": [
        ("panos", pan_globalprotect),
        ("panos", pan_ike),
        ("cisco_ios", ios_crypto_session),
        ("cisco_ios", ios_crypto_inv_spi),
    ],
    "wireless": [
        ("cisco_wlc", wlc_client_run),
        ("cisco_wlc", wlc_client_deauth),
        ("cisco_wlc", wlc_ap_join),
        ("cisco_wlc", wlc_apmgr),
        ("cisco_wlc", wlc_ap_heartbeat),
    ],
    "noise": [
        ("cisco_ios", ntp_sync),
        ("cisco_wlc", partial(ntp_sync, kind="wlc")),
        ("cisco_ios", cdp_native_vlan),
        ("cisco_ios", ios_login_success),
        ("cisco_ise", ise_5200),
        ("cisco_ise", ise_profiler),
        ("cisco_ios", ios_session_success),
        ("cisco_ise", ise_coa),
        ("cisco_ise", ise_accounting),
        ("cisco_wlc", wlc_webui_login),
        ("panos", pan_traffic),
        ("panos", pan_traffic),
        ("cisco_ios", ios_logging_host),
        ("panos", pan_scheduled_job),
    ],
}

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def render(template: str, fields: dict[str, Field]) -> tuple[str, list[dict[str, Any]]]:
    """Fill ``{name}`` placeholders and record entity character spans as we go.

    Spans are computed while the string is built, so ``text[start:end]`` is exact by
    construction — no regex re-discovery that could drift out of sync.
    """
    parts: list[str] = []
    entities: list[dict[str, Any]] = []
    length, pos = 0, 0
    for m in _PLACEHOLDER.finditer(template):
        literal = template[pos : m.start()]
        parts.append(literal)
        length += len(literal)
        value = fields[m.group(1)]
        if isinstance(value, tuple):
            value, etype = value
            if value:
                entities.append({"start": length, "end": length + len(value), "type": etype})
        parts.append(value)
        length += len(value)
        pos = m.end()
    parts.append(template[pos:])
    return "".join(parts), entities


def template_name(fn: Template) -> str:
    """Stable template id, e.g. ``ios_config_i`` or ``ios_config_i[wlc]`` for a partial."""
    if isinstance(fn, partial):
        return f"{fn.func.__name__}[{','.join(map(str, fn.keywords.values()))}]"
    return fn.__name__


def generate_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate ``data.num_samples`` unique labeled rows, balanced across labels (±1)."""
    rng = random.Random(cfg["seed"])
    labels: list[str] = cfg["labels"]
    vendors = set(cfg["data"]["vendors"])
    n_total = cfg["data"]["num_samples"]
    base, extra = divmod(n_total, len(labels))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, label in enumerate(labels):
        pool = [(v, t) for v, t in TEMPLATES[label] if v in vendors]
        if not pool:
            raise ValueError(f"No templates for label {label!r} with vendors {sorted(vendors)}")
        target = base + (1 if i < extra else 0)
        count, attempts = 0, 0
        while count < target:
            attempts += 1
            if attempts > target * 50:
                raise RuntimeError(f"Could not generate enough unique rows for {label!r}")
            vendor, template_fn = rng.choice(pool)
            dt = start + timedelta(seconds=rng.randrange(365 * 86400))
            text, entities = render(*template_fn(rng, dt))
            if text in seen:  # avoid exact duplicates leaking across splits
                continue
            seen.add(text)
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "vendor": vendor,
                    "template": template_name(template_fn),
                    "entities": entities,
                }
            )
            count += 1
    rng.shuffle(rows)
    return rows


def build_features(cfg: dict[str, Any]) -> Features:
    """The dataset schema. ``label`` is a ClassLabel, so it is stored as an int but keeps its name."""
    return Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=cfg["labels"]),
            "vendor": Value("string"),
            "template": Value("string"),
            # A Python list in Features means "variable-length list of this struct".
            "entities": [{"start": Value("int32"), "end": Value("int32"), "type": Value("string")}],
        }
    )


def build_dataset_dict(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> DatasetDict:
    """Turn rows into stratified train / validation / test splits, plus two "unseen" splits.

    ``test_unseen`` holds every row from the templates in ``data.heldout_templates`` and
    ``validation_unseen`` every row from ``data.val_heldout_templates``. Those templates never
    appear in train. Scoring on them asks "did the model learn what an auth failure *means*,
    or just memorize our templates?" — and ``validation_unseen`` lets training pick its best
    checkpoint by that question instead of by memorization.
    """
    label2id, _ = label_maps(cfg)
    records = [{**r, "label": label2id[r["label"]]} for r in rows]
    ds = Dataset.from_list(records, features=build_features(cfg))

    vendors = set(cfg["data"]["vendors"])
    # only templates of the configured vendors can be held out (others are never generated)
    known = {template_name(fn) for pool in TEMPLATES.values() for v, fn in pool if v in vendors}
    test_held = set(cfg["data"].get("heldout_templates") or [])
    if overlap := test_held & set(cfg["data"].get("val_heldout_templates") or []):
        raise ValueError(f"Templates in both held-out lists: {sorted(overlap)}")
    unseen_splits: dict[str, Dataset] = {}
    for split_name, key in (
        ("test_unseen", "heldout_templates"),
        ("validation_unseen", "val_heldout_templates"),
    ):
        names = set(cfg["data"].get(key) or [])
        if unknown := names - known:
            raise ValueError(f"Unknown {key} (or vendor not configured): {sorted(unknown)}")
        if names:
            unseen_splits[split_name] = ds.filter(lambda ex, n=names: ex["template"] in n)
            ds = ds.filter(lambda ex, n=names: ex["template"] not in n)

    split = cfg["data"]["split"]
    holdout = split["val"] + split["test"]
    # stratify_by_column keeps each split's class proportions the same as the full set.
    # It only works on a ClassLabel column — one reason we declared `label` that way.
    first = ds.train_test_split(test_size=holdout, stratify_by_column="label", seed=cfg["seed"])
    second = first["test"].train_test_split(
        test_size=split["test"] / holdout, stratify_by_column="label", seed=cfg["seed"]
    )
    dsd = DatasetDict(train=first["train"], validation=second["train"], test=second["test"])
    dsd.update(unseen_splits)
    return dsd


def save(rows: list[dict[str, Any]], dsd: DatasetDict, cfg: dict[str, Any]) -> None:
    """Write the raw rows as JSONL (human-readable) and the DatasetDict as Arrow."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "logs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    dsd.save_to_disk(cfg["data"]["processed_dir"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic LogSense dataset.")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--show", type=int, default=10, help="Print N random training examples")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows = generate_rows(cfg)
    dsd = build_dataset_dict(rows, cfg)
    save(rows, dsd, cfg)

    names = dsd["train"].features["label"].names
    print(dsd)
    for split_name, split_ds in dsd.items():
        counts = Counter(names[i] for i in split_ds["label"])
        print(f"{split_name:<10}", dict(sorted(counts.items())))
    print(f"\nSaved to {cfg['data']['processed_dir']} and {cfg['data']['raw_dir']}/logs.jsonl\n")

    sample = dsd["train"].shuffle(seed=cfg["seed"]).select(range(min(args.show, len(dsd["train"]))))
    for ex in sample:
        ents = ", ".join(f"{e['type']}={ex['text'][e['start'] : e['end']]}" for e in ex["entities"])
        print(f"[{names[ex['label']]}] {ex['text']}\n    -> {ents or '(no entities)'}")


if __name__ == "__main__":
    main()
