
DROP TABLE IF EXISTS input.bill_metadata;

CREATE TABLE bill_metadata (
	bill_id VARCHAR,
	chamber VARCHAR,
	created_at TIMESTAMP,
	id VARCHAR,
	session VARCHAR,
	state VARCHAR(2),
	subjects VARCHAR,
	title VARCHAR,
	type VARCHAR,
	updated_at TIMESTAMP
);
