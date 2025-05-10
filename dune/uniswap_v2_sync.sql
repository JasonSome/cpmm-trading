SELECT to_char(evt_block_time, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS ts, *
FROM uniswap_v2."Pair_evt_Sync"
WHERE contract_address = '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
AND evt_block_time >= '2020-02-21'::date
AND evt_block_time < '2020-03-01'::date
ORDER BY evt_block_number, evt_index ASC


SELECT date_trunc('year', evt_block_time) AS year, count(*)
FROM uniswap_v2."Pair_evt_Sync" s
WHERE contract_address = '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
GROUP BY 1
ORDER BY 1 ASC

SELECT
minute,
latest_reserves[3] AS reserve0,
latest_reserves[4] AS reserve1
FROM
(SELECT date_trunc('minute', evt_block_time) AS minute,
        (SELECT MAX(ARRAY[evt_block_number, evt_index, reserve0, reserve1])) AS latest_reserves
         FROM uniswap_v2."Pair_evt_Sync"
         WHERE contract_address =  '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
         GROUP BY 1) AS day_reserves
ORDER BY 1 ASC
