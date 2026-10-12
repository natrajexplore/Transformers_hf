---
license: mit
base_model: distilbert-base-uncased
library_name: transformers
pipeline_tag: token-classification
language: en
tags:
  - syslog
  - ner
  - network-security
  - cisco
  - palo-alto
model-index:
  - name: logsense-ner
    results:
      - task: {type: token-classification}
        dataset: {name: LogSense synthetic syslog (test_unseen), type: synthetic}
        metrics:
          - {type: f1, name: entity micro-F1, value: 0.863}
---

# LogSense NER

DistilBERT fine-tuned for token classification (BIO tags) to extract entities from syslog
lines: `IP`, `USER`, `INTERFACE`, `MAC`, `RULE`, `VLAN`. Covers Cisco IOS/IOS-XE, ISE,
Catalyst 9800 WLC, and PAN-OS formats.

```python
from transformers import pipeline

ner = pipeline(
    "token-classification", model="natrajexplore/logsense-ner", aggregation_strategy="first"
)
ner("Login failed [user: user07] [Source: 10.1.2.3] on GigabitEthernet1/0/3")
```

## Training data

~1.5k **synthetic, anonymized** log lines from 53 templates with exact character spans for
each entity (RFC 5737 / RFC 1918 IPs, fake users/hostnames/MACs). Spans were converted to
BIO labels on the first subword of each word (`word_ids()`), other subwords set to -100.
3 epochs, lr 5e-5, max 256 tokens; best checkpoint chosen on held-out-template entity F1.

## Evaluation (seqeval, entity level)

`test` (templates seen in training): micro-F1 **1.000** for every type.

`test_unseen` (7 templates never seen in training):

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| USER | 1.000 | 1.000 | 1.000 | 90 |
| INTERFACE | 0.997 | 1.000 | 0.998 | 295 |
| IP | 0.928 | 0.928 | 0.928 | 486 |
| VLAN | 0.719 | 0.479 | 0.575 | 48 |
| MAC | 0.276 | 1.000 | 0.432 | 73 |
| **micro avg** | 0.797 | 0.940 | **0.863** | 992 |

`RULE` has no examples in the unseen templates, so it is only measured on `test`.

## Limitations

* Bare numbers are the weak spot: port numbers after an IP (`198.51.100.7:22`) and VLAN IDs
  get confused with IP/MAC/VLAN, which is why MAC precision and VLAN recall are low.
* Only six entity types; hostnames, SSIDs, threat names, and session IDs are not extracted.
* Trained only on synthetic logs; formats the generator never produced will be weaker.
