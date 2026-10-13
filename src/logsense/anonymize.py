"""Best-effort anonymizer for real syslog lines before they are labeled or shared.

Replaces, consistently within one call (the same real value always gets the same fake value,
so "the same user failed 5 times" stays visible):

* IPv4 — public addresses -> RFC 5737 documentation ranges; private (RFC 1918 / CGNAT) ->
  10.x.y.z. Already-documentation, loopback, multicast, masks and other reserved
  addresses are kept: they don't identify anything.
* MAC addresses in Cisco dotted, colon and dash formats -> locally administered 02:00:5e:...
  in the same format and case.
* Usernames in the common Cisco / ISE / PAN-OS / C9800 positions -> ``userNN``; emails ->
  ``userNN@example.com``.
* Device hostnames (the syslog header, AP names, ``Hostname=`` style fields, FQDNs) ->
  ``corp-host-NN`` / ``corp-ap-NN`` / ``hostNN.example.com``; PAN-OS serial numbers.

It is pattern-based, so it can miss values in unusual positions (free-text descriptions,
IPv6, custom fields). Always read the result before sharing it.
"""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from typing import Any

DOC_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
CGNAT = ipaddress.ip_network("100.64.0.0/10")
KEEP_USERS = {"admin", "root", "enable", "console", "unknown", "n/a", "none", "system", "-"}

IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
MAC_DOTTED = re.compile(r"\b[0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\b")
MAC_SEP = re.compile(r"\b[0-9a-fA-F]{2}([:-])(?:[0-9a-fA-F]{2}\1){4}[0-9a-fA-F]{2}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# Syslog header: optional <PRI>, "Mon dd hh:mm:ss", then the sending device's hostname.
HEADER_HOST = re.compile(r"^(<\d+>)?([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+)([^\s:]+)")
PAN_SERIAL = re.compile(r"(\s1,\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2},)(\d{9,15})(,)")
USER_PATTERNS = [
    re.compile(r"\[user: ([^\]\s]+)\]"),  # IOS LOGIN_FAILED / LOGIN_SUCCESS
    re.compile(r"\b(?:UserName|User-Name|Username|username)=([^,\s]+)"),  # ISE
    re.compile(r"\bUser name: ([^,\s]+)"),  # PAN-OS GlobalProtect SYSTEM
    re.compile(r"\b[Uu]ser '([^']+)'"),  # IOS SSH, PAN-OS auth, C9800 web UI
    re.compile(r"\bUsername entry \(([^)]+)\)"),  # C9800 CLIENT_ORCH_LOG
    re.compile(r"\bconsole by ([\w.\\-]+) on\b"),  # IOS SYS-5-CONFIG_I
    re.compile(r"\buser ([\w.\\-]+) (?:logged|authenticated|failed)\b", re.IGNORECASE),
]
HOST_PATTERNS = [
    (re.compile(r"\bAP (?:Name: ?)?([A-Za-z][\w.-]*) \("), "ap"),  # C9800 "AP corp-ap-46 (mac)"
    (re.compile(r"\b(?:[Hh]ost[Nn]ame|NAS-Identifier|DeviceName)=([^,\s]+)"), "host"),
    (re.compile(r"\bwith ([A-Za-z][\w.-]*) (?=[A-Z][a-z]*Ethernet|Gi|Te|Fa)"), "host"),  # CDP peer
]
FQDN = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|local|corp|lan|internal|intra)\b", re.IGNORECASE
)


class Anonymizer:
    """Holds the real -> fake mappings, so one batch of lines is anonymized consistently."""

    def __init__(self) -> None:
        self.maps: dict[str, dict[str, str]] = {}
        self.counts: Counter[str] = Counter()

    def _fake(self, kind: str, real: str, make) -> str:
        table = self.maps.setdefault(kind, {})
        if real not in table:
            table[real] = make(len(table) + 1)
        self.counts[kind] += 1
        return table[real]

    # -- IPs -----------------------------------------------------------------
    def _ip(self, m: re.Match) -> str:
        text = m.group(1)
        try:
            ip = ipaddress.IPv4Address(text)
        except ipaddress.AddressValueError:
            return text  # e.g. a version string like 999.1.2.3
        if (
            any(ip in n for n in DOC_NETS)
            or ip.is_loopback
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_link_local
            or ip.is_reserved
            or text.startswith("255.")  # netmasks
        ):
            return text
        if ip.is_private or ip in CGNAT:
            return self._fake("ip", text, lambda n: f"10.{n >> 16 & 255}.{n >> 8 & 255}.{n & 255}")
        nets = ["192.0.2", "198.51.100", "203.0.113"]
        return self._fake("ip", text, lambda n: f"{nets[(n - 1) // 254 % 3]}.{(n - 1) % 254 + 1}")

    # -- MACs ----------------------------------------------------------------
    def _mac_octets(self, real: str) -> list[str]:
        key = re.sub(r"[^0-9a-f]", "", real.lower())
        fake = self.maps.setdefault("mac", {})
        if key not in fake:
            n = len(fake) + 1
            fake[key] = f"02005e{n >> 16 & 0xFF:02x}{n >> 8 & 0xFF:02x}{n & 0xFF:02x}"
        self.counts["mac"] += 1
        h = fake[key]
        return [h[i : i + 2] for i in range(0, 12, 2)]

    def _mac_dotted(self, m: re.Match) -> str:
        o = self._mac_octets(m.group(0))
        out = f"{o[0]}{o[1]}.{o[2]}{o[3]}.{o[4]}{o[5]}"
        return out.upper() if m.group(0).isupper() else out

    def _mac_sep(self, m: re.Match) -> str:
        o = self._mac_octets(m.group(0))
        out = m.group(1).join(o)
        return out.upper() if m.group(0).upper() == m.group(0) else out

    # -- names ---------------------------------------------------------------
    def _user(self, real: str) -> str:
        if real.lower() in KEEP_USERS:
            return real
        return self._fake("user", real, lambda n: f"user{n:02d}")

    @staticmethod
    def _sub_group(pattern: re.Pattern, line: str, fn) -> str:
        """Replace only group 1 of each match, keeping the surrounding text."""

        def repl(m: re.Match) -> str:
            s, e = m.span(1)
            base = m.start(0)
            whole = m.group(0)
            return whole[: s - base] + fn(m.group(1)) + whole[e - base :]

        return pattern.sub(repl, line)

    def _fqdn(self, m: re.Match) -> str:
        name = m.group(0).lower()
        if name == "example.com" or name.endswith(".example.com"):  # already a placeholder
            return m.group(0)
        return self._fake("fqdn", name, lambda n: f"host{n:02d}.example.com")

    def line(self, line: str) -> str:
        line = EMAIL.sub(
            lambda m: self._fake("email", m.group(0), lambda n: f"user{n:02d}@example.com"), line
        )
        line = PAN_SERIAL.sub(
            lambda m: (
                m.group(1)
                + self._fake("serial", m.group(2), lambda n: f"0128010{n:05d}")
                + m.group(3)
            ),
            line,
        )
        line = MAC_DOTTED.sub(self._mac_dotted, line)
        line = MAC_SEP.sub(self._mac_sep, line)
        line = IPV4.sub(self._ip, line)
        for pattern in USER_PATTERNS:
            line = self._sub_group(pattern, line, self._user)
        line = HEADER_HOST.sub(
            lambda m: (
                (m.group(1) or "")
                + m.group(2)
                + self._fake("host", m.group(3), lambda n: f"corp-host-{n:02d}")
            ),
            line,
        )
        for pattern, kind in HOST_PATTERNS:
            line = self._sub_group(
                pattern,
                line,
                lambda v, kind=kind: self._fake(kind, v, lambda n: f"corp-{kind}-{n:02d}"),
            )
        line = FQDN.sub(self._fqdn, line)
        return line


def anonymize_lines(lines: list[str]) -> dict[str, Any]:
    """Anonymize a batch consistently. Returns the new lines and how many values were replaced."""
    anon = Anonymizer()
    out = [anon.line(ln) for ln in lines]
    return {
        "lines": out,
        "replaced": dict(anon.counts),
        "distinct": {kind: len(table) for kind, table in anon.maps.items()},
    }
