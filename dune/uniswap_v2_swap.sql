SELECT to_char(evt_block_time, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS ts, *
FROM uniswap_v2."Pair_evt_Swap"
WHERE contract_address = '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
AND evt_block_time >= '2020-02-21'::date
AND evt_block_time < '2020-03-01'::date
ORDER BY evt_block_number, evt_index ASC

SELECT date_trunc('year', evt_block_time) AS year, count(*)
FROM uniswap_v2."Pair_evt_Swap" s
WHERE contract_address = '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
GROUP BY 1
ORDER BY 1 ASC

SELECT
date_trunc('minute', evt_block_time) AS minute,
SUM("amount0In") as "amount0In",
SUM("amount1In") as "amount1In",
SUM("amount0Out") as "amount0Out",
SUM("amount1Out") as "amount1Out"
FROM uniswap_v2."Pair_evt_Swap"
WHERE contract_address =  '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
GROUP BY 1
ORDER BY 1 ASC





SELECT
b.number, count(*) as swap_count
FROM uniswap_v2."Pair_evt_Swap" s
FULL OUTER JOIN ethereum.blocks b
ON s.evt_block_number=b.number
WHERE s.contract_address = '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
GROUP BY 1
ORDER BY 1 ASC
LIMIT 10


SELECT
evt_block_number, count(*) as swap_count
FROM uniswap_v2."Pair_evt_Swap" s
WHERE contract_address = '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
GROUP BY 1
ORDER BY 1 ASC
LIMIT 10
