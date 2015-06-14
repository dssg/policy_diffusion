
DROP TABLE IF EXISTS bill_metadata;

CREATE TABLE bill_metadata (
	bill_id VARCHAR,
	title VARCHAR,
	alternate_titles JSON,
	versions VARCHAR,
	subjects VARCHAR,
	scraped_subjects VARCHAR,
	type VARCHAR,
	level VARCHAR,
	sponsors JSON,
	actions JSON,
	action_dates JSON,
	documents JSON,
	votes JSON,
	leg_id VARCHAR,
	state CHAR(2),
	chamber VARCHAR,
	session VARCHAR,
	all_ids VARCHAR,
	created_at TIMESTAMP WITHOUT TIME ZONE,
	updated_at TIMESTAMP WITHOUT TIME ZONE
);
