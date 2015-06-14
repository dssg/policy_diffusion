
DROP TABLE IF EXISTS state_metadata;

CREATE TABLE state_metadata (
	name VARCHAR(20),
	abbreviation VARCHAR(2),
	lower_chamber_name VARCHAR(10),
	lower_chamber_title VARCHAR(15),
	upper_chamber_name VARCHAR(10),
	upper_chamber_title VARCHAR(15),
	feature_flags VARCHAR(50)
);
