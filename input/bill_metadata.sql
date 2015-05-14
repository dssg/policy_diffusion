
DROP TABLE IF EXISTS input.bill_metadata;

CREATE TABLE input.bill_metadata (
	bill_id VARCHAR(10),
	chamber VARCHAR(10),
	created_at TIMESTAMP,
	id VARCHAR(20),
	session VARCHAR(8),
	state VARCHAR(2),
	subjects VARCHAR,
	title VARCHAR,
	type VARCHAR(10),
	updated_at TIMESTAMP
);
