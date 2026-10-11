grid = LOAD '/data/grid_demand.csv' USING PigStorage(',') AS (
    demand_date:chararray,
    state:chararray,
    max_demand_met:float,
    shortage_peak:float,
    energy_met:float,
    drawl_schedule:float,
    od_ud:float,
    max_od:float,
    energy_shortage:float
);

grid_clean = FILTER grid BY state != 'State';

grid_clean = FILTER grid_clean BY
    max_demand_met IS NOT NULL AND
    energy_met IS NOT NULL AND
    drawl_schedule IS NOT NULL;

grid_clean = FILTER grid_clean BY max_demand_met > 0;

STORE grid_clean INTO '/tmp/grid_demand_clean' USING PigStorage(',');
