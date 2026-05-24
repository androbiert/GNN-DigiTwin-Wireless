# Extreme Value EDA

- Flow rows analyzed: 3,562,752
- Delay configs analyzed: 192
- Delay scenarios discovered: SC01, SC02
- Median delay: 8.673 ms
- p95 delay: 23890.100 ms
- p99 delay: 70143.402 ms
- Max delay: 118836.998 ms
- IQR upper fence: 9.975 ms
- Extreme threshold used: p99 = 70143.402 ms
- Share above p99 threshold: 1.001%
- Delay skewness proxy (mean/median): 358.48x

## Interpretation

The delay distribution is strongly right-skewed, with a very heavy upper tail.

## Heaviest Tail by Scenario

- SC01: count=646,784, mean=13033.895 ms, p95=72851.782 ms, p99=100211.998 ms, max=118836.998 ms
- SC02: count=2,915,968, mean=907.797 ms, p95=20.568 ms, p99=34011.002 ms, max=47015.999 ms

## Heaviest Tail by Scheduler

- MAXCI_MB: count=800,720, mean=2429.854 ms, p95=821.926 ms, p99=85611.000 ms, max=118836.998 ms
- PF: count=920,768, mean=2316.194 ms, p95=17698.400 ms, p99=68157.840 ms, max=104181.000 ms
- MAXCI: count=920,432, mean=2502.196 ms, p95=14896.500 ms, p99=63648.602 ms, max=110837.997 ms
- DRR: count=920,832, mean=5099.531 ms, p95=34012.501 ms, p99=54949.501 ms, max=113836.998 ms

## Heaviest Tail by Queue Size

- 10MiB: count=891,420, mean=6157.090 ms, p95=47312.698 ms, p99=90580.597 ms, max=118836.998 ms
- 2MiB: count=891,420, mean=3558.256 ms, p95=25186.100 ms, p99=57324.200 ms, max=118836.998 ms
- 100KiB: count=889,956, mean=1486.224 ms, p95=4294.470 ms, p99=40012.501 ms, max=118005.997 ms
- 50KiB: count=889,956, mean=1229.381 ms, p95=2411.990 ms, p99=36014.500 ms, max=117654.999 ms

## Configurations Contributing Most Extreme Delays

- 20)SC01-P=0.01W-S=MAXCI_MB-Q=10MiB (SC01, MAXCI_MB, 0.01W, 10MiB): extreme_share=28.03%, p99=109349.998 ms, max=118836.998 ms
- 44)SC01-P=0.1W-S=MAXCI_MB-Q=10MiB (SC01, MAXCI_MB, 0.1W, 10MiB): extreme_share=28.03%, p99=109349.998 ms, max=118836.998 ms
- 68)SC01-P=0.5W-S=MAXCI_MB-Q=10MiB (SC01, MAXCI_MB, 0.5W, 10MiB): extreme_share=28.03%, p99=109349.998 ms, max=118836.998 ms
- 92)SC01-P=2W-S=MAXCI_MB-Q=10MiB (SC01, MAXCI_MB, 2W, 10MiB): extreme_share=28.03%, p99=109349.998 ms, max=118836.998 ms
- 18)SC01-P=0.01W-S=MAXCI_MB-Q=100KiB (SC01, MAXCI_MB, 0.01W, 100KiB): extreme_share=22.51%, p99=113948.997 ms, max=118005.997 ms
- 42)SC01-P=0.1W-S=MAXCI_MB-Q=100KiB (SC01, MAXCI_MB, 0.1W, 100KiB): extreme_share=22.51%, p99=113948.997 ms, max=118005.997 ms
- 66)SC01-P=0.5W-S=MAXCI_MB-Q=100KiB (SC01, MAXCI_MB, 0.5W, 100KiB): extreme_share=22.51%, p99=113948.997 ms, max=118005.997 ms
- 90)SC01-P=2W-S=MAXCI_MB-Q=100KiB (SC01, MAXCI_MB, 2W, 100KiB): extreme_share=22.51%, p99=113948.997 ms, max=118005.997 ms
- 19)SC01-P=0.01W-S=MAXCI_MB-Q=2MiB (SC01, MAXCI_MB, 0.01W, 2MiB): extreme_share=18.71%, p99=109349.998 ms, max=118836.998 ms
- 43)SC01-P=0.1W-S=MAXCI_MB-Q=2MiB (SC01, MAXCI_MB, 0.1W, 2MiB): extreme_share=18.71%, p99=109349.998 ms, max=118836.998 ms
- 67)SC01-P=0.5W-S=MAXCI_MB-Q=2MiB (SC01, MAXCI_MB, 0.5W, 2MiB): extreme_share=18.71%, p99=109349.998 ms, max=118836.998 ms
- 91)SC01-P=2W-S=MAXCI_MB-Q=2MiB (SC01, MAXCI_MB, 2W, 2MiB): extreme_share=18.71%, p99=109349.998 ms, max=118836.998 ms
- 04)SC01-P=0.01W-S=PF-Q=10MiB (SC01, PF, 0.01W, 10MiB): extreme_share=18.07%, p99=102346.803 ms, max=104181.000 ms
- 28)SC01-P=0.1W-S=PF-Q=10MiB (SC01, PF, 0.1W, 10MiB): extreme_share=18.07%, p99=102346.803 ms, max=104181.000 ms
- 52)SC01-P=0.5W-S=PF-Q=10MiB (SC01, PF, 0.5W, 10MiB): extreme_share=18.07%, p99=102346.803 ms, max=104181.000 ms

## Robust Feature Outlier Rates

- offered_load_bps: 44.720%
- delivery_ratio: 17.438%
- speed: 12.908%
- sinr_ul: 0.261%
- sinr_dl: 0.016%
- packet_loss: 0.000%
- harq_error_rate: 0.000%
- harq_tx_attempts: 0.000%
- distance: 0.000%
- queue_bytes: 0.000%
