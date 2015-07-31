DROP TABLE IF EXISTS opensecrets.candidates;

CREATE TABLE opensecrets.candidates (
	cycle INTEGER NOT NULL, 
	fec_candidate_id VARCHAR(9) NOT NULL, 
	candidate_id VARCHAR(9) NOT NULL, 
	first_last_party VARCHAR(38) NOT NULL, 
	party VARCHAR(7) NOT NULL, 
	office_sought VARCHAR(4), 
	office_held VARCHAR(4), 
	currently_running BOOLEAN, 
	 VARCHAR(4), 
	"RL" VARCHAR(4)
);
