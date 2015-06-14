
DROP TABLE IF EXISTS committees;

CREATE TABLE committees (
	id VARCHAR,
	state VARCHAR(2),
	chamber VARCHAR(10),
	committee VARCHAR,
	subcommittee VARCHAR,
	members JSON,
	sources VARCHAR,
	parent_id VARCHAR(10),
	created_at TIMESTAMP WITHOUT TIME ZONE,
	updated_at TIMESTAMP WITHOUT TIME ZONE,
	all_ids VARCHAR,
	level VARCHAR(5)
);
