
DROP TABLE IF EXISTS committee_metadata;

CREATE TABLE committee_metadata (
	id VARCHAR,
	state VARCHAR(2),
	chamber VARCHAR(10),
	committee VARCHAR,
	subcommittee VARCHAR,
	members VARCHAR,
	sources JSON,
	parent_id VARCHAR(10),
	created_at TIMESTAMP WITHOUT TIME ZONE,
	updated_at TIMESTAMP WITHOUT TIME ZONE,
	all_ids VARCHAR,
	level VARCHAR(5)
);
