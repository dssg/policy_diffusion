
DROP TABLE IF EXISTS input.state_metadata;

CREATE TABLE input.state_metadata (
	abbreviation VARCHAR(2),
	lower_chamber_name VARCHAR(10),
	lower_chamber_title VARCHAR(15),
	upper_chamber_name VARCHAR(10),
	upper_chamber_title VARCHAR(15),
	feature_flags VARCHAR(50),
	name VARCHAR(20)
);

\COPY input.state_metadata FROM 'data/input/state_metadata.csv' WITH CSV;

ALTER TABLE input.state_metadata ADD COLUMN bills_identified BOOLEAN;
