
DROP TABLE IF EXISTS legislators;

CREATE TABLE legislators (
	id VARCHAR,
	votesmart_id VARCHAR,
	transparencydata_id VARCHAR,
	first_name VARCHAR,
	middle_name VARCHAR,
	last_name VARCHAR,
	suffixes VARCHAR,
	full_name VARCHAR,
	party VARCHAR,
	active BOOLEAN,
	url VARCHAR,
	photo_url VARCHAR,
	office_address VARCHAR,
	office_phone VARCHAR,
	leg_id VARCHAR,
	chamber VARCHAR,
	district VARCHAR,
	state VARCHAR,
	offices JSON,
	email VARCHAR,
	roles JSON,
	old_roles JSON,
	all_legislative_ids VARCHAR,
	level VARCHAR,
	sources JSON,
	created_at TIMESTAMP WITHOUT TIME ZONE,
	updated_at TIMESTAMP WITHOUT TIME ZONE
);
