from logsense.anonymize import anonymize_lines

IOS = (
    "Oct 10 08:02:11 sw-bldg4 4411: *Oct 10 08:02:11.101 UTC: %SEC_LOGIN-4-LOGIN_FAILED: "
    "Login failed [user: jsmith] [Source: 172.20.4.9] [localport: 22] at 08:02:11"
)
ISE = (
    "Oct 10 08:05:00 ise-prod-01 CISE_Failed_Attempts 1 1 0 2026-10-10 08:05:00.120 +00:00 2 5400 "
    "NOTICE Failed-Attempt: Authentication failed, Device IP Address=10.9.8.7, UserName=jsmith, "
    "Calling-Station-ID=A4-5E-60-11-22-33, mask 255.255.255.0, dns 8.8.8.8, doc 192.0.2.5"
)
WLC = (
    "Oct 10 09:20:00 wlc-hq: *Oct 10 09:20:00.065: %APMGR_TRACE_MESSAGE-3-AP_EVENT: Chassis 1 "
    "R0/0: wncd: AP AP-Lobby-01 (a45e.6011.2233): Radar detected, portal vpn.acme.com"
)


def test_values_are_replaced_consistently():
    out = anonymize_lines([IOS, IOS.replace("08:02:11", "08:02:12"), ISE])["lines"]
    for real in ("sw-bldg4", "jsmith", "172.20.4.9", "ise-prod-01", "10.9.8.7", "8.8.8.8"):
        assert all(real not in line for line in out), real
    assert out[0].split()[3] == out[1].split()[3] == "corp-host-01"  # same host, same fake
    assert "[user: user01]" in out[0] and "UserName=user01" in out[2]  # same user across vendors


def test_harmless_values_are_kept():
    out = anonymize_lines([ISE])["lines"][0]
    assert "255.255.255.0" in out and "192.0.2.5" in out  # netmask, documentation IP
    assert "Device IP Address=10." in out  # private stays private
    assert "dns 192.0.2." in out or "dns 198.51.100." in out or "dns 203.0.113." in out


def test_mac_format_and_identity_preserved():
    r = anonymize_lines([ISE, WLC])
    ise, wlc = r["lines"]
    assert "Calling-Station-ID=02-00-5E-00-00-01" in ise  # dash format, upper case kept
    assert "(0200.5e00.0001)" in wlc  # same MAC in dotted format -> same fake
    assert "AP corp-ap-01 (" in wlc and "host01.example.com" in wlc
    assert r["distinct"]["mac"] == 1


def test_counts_reported():
    r = anonymize_lines([IOS])
    assert r["replaced"]["user"] == 1 and r["replaced"]["ip"] == 1 and r["replaced"]["host"] == 1
