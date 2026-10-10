---
license: apache-2.0
base_model: distilbert-base-uncased
library_name: transformers
pipeline_tag: text-classification
language: en
tags:
  - syslog
  - network-security
  - cisco
  - palo-alto
  - log-classification
model-index:
  - name: logsense-classifier
    results:
      - task: {type: text-classification}
        dataset: {name: LogSense synthetic syslog (test_unseen), type: synthetic}
        metrics:
          - {type: accuracy, value: 0.51}
          - {type: f1, name: macro-F1, value: 0.475}
---

# LogSense classifier

DistilBERT fine-tuned to sort one syslog line into a triage category:
`auth_failure`, `config_change`, `link_event`, `threat`, `vpn`, `wireless`, `noise`.
Covers Cisco IOS/IOS-XE, Cisco ISE, Catalyst 9800 WLC, and Palo Alto PAN-OS formats.

```python
from transformers import pipeline

clf = pipeline("text-classification", model="natrajexplore/logsense-classifier")
clf("%LINK-3-UPDOWN: Interface GigabitEthernet1/0/7, changed state to down")
```

## Training data

~1.5k **synthetic, anonymized** log lines from 53 message templates (RFC 5737 / RFC 1918 IPs,
fake users and hostnames). Some message formats follow examples shown in Cisco Live 2026
sessions (BRKEWN-3628, BRKSEC-2347). No real logs were used. 3 epochs, lr 2e-5,
max 128 tokens, best checkpoint chosen on held-out-template validation macro-F1.

## Evaluation — read this before trusting the model

| Split | What it measures | Accuracy | Macro-F1 |
|---|---|---|---|
| `test` | new lines from templates seen in training | 1.000 | 1.000 |
| `test_unseen` | lines from 7 templates never seen in training | 0.510 | **0.475** |

Baselines on `test_unseen`: TF-IDF + logistic regression **0.632** macro-F1 (better than this
model), zero-shot `bart-large-mnli` 0.369.

Per-class F1 on `test_unseen`: config_change 1.00, wireless 1.00, link_event 0.91,
auth_failure 0.42, threat 0.00, vpn 0.00, noise 0.00.

## Limitations

* The perfect `test` score reflects template memorization, not real-world accuracy.
* On unfamiliar formats the model leans on vendor/format cues: e.g. an IOS IPS signature is
  not recognized as `threat`, and PAN-OS IKE messages are mistaken for `auth_failure`.
* Not evaluated on real production logs. Use `python -m logsense.evaluate_file` on your own
  labeled, anonymized logs before relying on it.
* Triage aid only — not a replacement for a SIEM or an analyst.
