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
* A PAN-OS TRAFFIC log is ``noise`` even when the action is ``deny`` (routine policy
  enforcement). Only THREAT logs are ``threat``.
* Any GlobalProtect log is ``vpn``, including GlobalProtect authentication failures.
* An interface up/down on the WLC (its uplinks) is ``link_event``, not ``wireless``.
* Routine messages from the WLC (NTP sync) are ``noise``, not ``wireless``.

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
        "UserName={user}, Calling-Station-ID={mac}, "
        "FailureReason=11036 The Message-Authenticator RADIUS attribute is invalid"
    )
    return tpl, {
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


# (vendor, template) pairs per label. Vendors are filtered by config `data.vendors`.
TEMPLATES: dict[str, list[tuple[str, Template]]] = {
    "auth_failure": [
        ("cisco_ios", ios_login_failed),
        ("cisco_ios", ios_ssh_failed),
        ("cisco_ise", ise_5400),
        ("cisco_ise", ise_5405),
        ("panos", pan_auth_fail),
    ],
    "config_change": [
        ("cisco_ios", ios_config_i),
        ("cisco_ios", ios_parser_cmd),
        ("cisco_wlc", partial(ios_config_i, kind="wlc")),
        ("cisco_ise", ise_config),
        ("panos", pan_config),
    ],
    "link_event": [
        ("cisco_ios", ios_link_updown),
        ("cisco_ios", ios_lineproto),
        ("cisco_ios", ios_admin_down),
        ("cisco_wlc", partial(ios_link_updown, kind="wlc")),
        ("panos", pan_link_change),
    ],
    "threat": [
        ("panos", pan_threat),
        ("panos", pan_threat),
        ("cisco_ios", ios_ips_signature),
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
    ],
    "noise": [
        ("cisco_ios", ntp_sync),
        ("cisco_wlc", partial(ntp_sync, kind="wlc")),
        ("cisco_ios", cdp_native_vlan),
        ("cisco_ios", ios_login_success),
        ("cisco_ise", ise_5200),
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
            rows.append({"text": text, "label": label, "vendor": vendor, "entities": entities})
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
            # A Python list in Features means "variable-length list of this struct".
            "entities": [{"start": Value("int32"), "end": Value("int32"), "type": Value("string")}],
        }
    )


def build_dataset_dict(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> DatasetDict:
    """Turn rows into a stratified train / validation / test DatasetDict."""
    label2id, _ = label_maps(cfg)
    records = [{**r, "label": label2id[r["label"]]} for r in rows]
    ds = Dataset.from_list(records, features=build_features(cfg))

    split = cfg["data"]["split"]
    holdout = split["val"] + split["test"]
    # stratify_by_column keeps each split's class proportions the same as the full set.
    # It only works on a ClassLabel column — one reason we declared `label` that way.
    first = ds.train_test_split(test_size=holdout, stratify_by_column="label", seed=cfg["seed"])
    second = first["test"].train_test_split(
        test_size=split["test"] / holdout, stratify_by_column="label", seed=cfg["seed"]
    )
    return DatasetDict(train=first["train"], validation=second["train"], test=second["test"])


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
