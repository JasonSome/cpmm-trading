SELECT to_char(evt_block_time, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS ts, *
FROM uniswap_v2."Pair_evt_Burn"
WHERE contract_address = '\xb4e16d0168e52d35cacd2c6185b44281ec28c9dc'
ORDER BY evt_block_number, evt_index ASC
