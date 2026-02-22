# Data Structure Report

**Dataset:** `cybersecurity_intrusion_data.csv`

## Shape

| Rows | Columns |
|------|---------|
| 9,537 | 11 |

## Sample (first 10 rows)

| session_id   |   network_packet_size | protocol_type   |   login_attempts |   session_duration | encryption_used   |   ip_reputation_score |   failed_logins | browser_type   |   unusual_time_access |   attack_detected |
|:-------------|----------------------:|:----------------|-----------------:|-------------------:|:------------------|----------------------:|----------------:|:---------------|----------------------:|------------------:|
| SID_00001    |                   599 | TCP             |                4 |           492.983  | DES               |             0.606818  |               1 | Edge           |                     0 |                 1 |
| SID_00002    |                   472 | TCP             |                3 |          1558      | DES               |             0.301569  |               0 | Firefox        |                     0 |                 0 |
| SID_00003    |                   629 | TCP             |                3 |            75.0443 | DES               |             0.739164  |               2 | Chrome         |                     0 |                 1 |
| SID_00004    |                   804 | UDP             |                4 |           601.249  | DES               |             0.123267  |               0 | Unknown        |                     0 |                 1 |
| SID_00005    |                   453 | TCP             |                5 |           532.541  | AES               |             0.0548739 |               1 | Firefox        |                     0 |                 0 |
| SID_00006    |                   453 | UDP             |                5 |           380.472  | AES               |             0.422486  |               2 | Chrome         |                     1 |                 0 |
| SID_00007    |                   815 | ICMP            |                4 |           728.107  | AES               |             0.413772  |               1 | Chrome         |                     0 |                 1 |
| SID_00008    |                   653 | TCP             |                3 |            12.5999 | DES               |             0.0977194 |               3 | Chrome         |                     1 |                 1 |
| SID_00009    |                   406 | TCP             |                2 |           542.559  | nan               |             0.29458   |               0 | Chrome         |                     1 |                 0 |
| SID_00010    |                   608 | UDP             |                6 |           531.944  | nan               |             0.424117  |               1 | Chrome         |                     0 |                 0 |

## Column Summary

|                     | dtype   |   nulls |   null_% |   unique |
|:--------------------|:--------|--------:|---------:|---------:|
| session_id          | str     |       0 |     0    |     9537 |
| network_packet_size | int64   |       0 |     0    |      959 |
| protocol_type       | str     |       0 |     0    |        3 |
| login_attempts      | int64   |       0 |     0    |       13 |
| session_duration    | float64 |       0 |     0    |     9532 |
| encryption_used     | str     |    1966 |    20.61 |        2 |
| ip_reputation_score | float64 |       0 |     0    |     9537 |
| failed_logins       | int64   |       0 |     0    |        6 |
| browser_type        | str     |       0 |     0    |        5 |
| unusual_time_access | int64   |       0 |     0    |        2 |
| attack_detected     | int64   |       0 |     0    |        2 |

## Numeric Columns — Descriptive Stats

|                     |     count |     mean |      std |     min |      25% |      50% |       75% |       max |   skew |
|:--------------------|----------:|---------:|---------:|--------:|---------:|---------:|----------:|----------:|-------:|
| network_packet_size | 9537.0000 | 500.4306 | 198.3794 | 64.0000 | 365.0000 | 499.0000 |  635.0000 | 1285.0000 | 0.0960 |
| login_attempts      | 9537.0000 |   4.0321 |   1.9630 |  1.0000 |   3.0000 |   4.0000 |    5.0000 |   13.0000 | 0.5963 |
| session_duration    | 9537.0000 | 792.7453 | 786.5601 |  0.5000 | 231.9530 | 556.2775 | 1105.3806 | 7190.3922 | 2.0846 |
| ip_reputation_score | 9537.0000 |   0.3313 |   0.1772 |  0.0025 |   0.1919 |   0.3148 |    0.4534 |    0.9243 | 0.4546 |
| failed_logins       | 9537.0000 |   1.5178 |   1.0340 |  0.0000 |   1.0000 |   1.0000 |    2.0000 |    5.0000 | 0.4064 |
| unusual_time_access | 9537.0000 |   0.1499 |   0.3570 |  0.0000 |   0.0000 |   0.0000 |    0.0000 |    1.0000 | 1.9613 |
| attack_detected     | 9537.0000 |   0.4471 |   0.4972 |  0.0000 |   0.0000 |   0.0000 |    1.0000 |    1.0000 | 0.2128 |

## Categorical Columns — Value Counts

### protocol_type

| protocol_type   |   count |     % |
|:----------------|--------:|------:|
| TCP             |    6624 | 69.46 |
| UDP             |    2406 | 25.23 |
| ICMP            |     507 |  5.32 |

### encryption_used

| encryption_used   |   count |     % |
|:------------------|--------:|------:|
| AES               |    4706 | 49.34 |
| DES               |    2865 | 30.04 |
| nan               |    1966 | 20.61 |

### browser_type

| browser_type   |   count |     % |
|:---------------|--------:|------:|
| Chrome         |    5137 | 53.86 |
| Firefox        |    1944 | 20.38 |
| Edge           |    1469 | 15.4  |
| Unknown        |     502 |  5.26 |
| Safari         |     485 |  5.09 |

## Target Variable: `attack_detected`

|   attack_detected |   count |     % |
|------------------:|--------:|------:|
|                 0 |    5273 | 55.29 |
|                 1 |    4264 | 44.71 |

**Imbalance ratio (majority / minority):** 1.24

## Duplicates

Duplicate rows: **0** (0.00%)
