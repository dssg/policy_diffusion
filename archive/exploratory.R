library('RPostgreSQL')
library('ggplot2')

db_info <- read.csv('policy_diffusion/default_profile', sep='=', header=F, quote='', stringsAsFactors=F)

# sessions
drv <- dbDriver('PostgreSQL')
con <- dbConnect(drv, user=db_info$V2[3], password=db_info$V2[4], 
                 dbname=db_info$V2[2], host=db_info$V2[1])

# number of governments
dbGetQuery(con, "SELECT COUNT(*) FROM (SELECT DISTINCT state FROM bill_metadata) AS a;")

# list the governments
dbGetQuery(con, "SELECT DISTINCT state FROM bill_metadata ORDER BY state;")

# number of sessions
dbGetQuery(con, "SELECT COUNT(*) FROM (SELECT DISTINCT state, session FROM bill_metadata ORDER BY state, session) AS a;")

# sessions
dbGetQuery(con, "SELECT DISTINCT state, session FROM bill_metadata ORDER BY session;")

# oldest session per government
dbGetQuery(con, "SELECT state, MIN(session) AS min_session FROM bill_metadata GROUP BY state ORDER BY state;")

# newest session per government
dbGetQuery(con, "SELECT state, MAX(session) AS max_session FROM bill_metadata GROUP BY state ORDER BY state;")

# bills and resolutions by government session
bills_and_resolutions <-
    dbGetQuery(con, "SELECT a.state, 
                            a.session,
                            a.bill_freq,
                            b.resolution_freq
                     FROM   (SELECT state, session, count(*) as bill_freq FROM bill_metadata WHERE type LIKE '%bill%' GROUP BY state, session) AS a,
                            (SELECT state, session, count(*) as resolution_freq FROM bill_metadata WHERE type LIKE '%resolution%' GROUP BY state, session) AS b
                     WHERE  a.state = b.state AND
                            a.session = b.session
                     ORDER BY bill_freq DESC;")

br_plt <- ggplot(bills_and_resolutions, aes(bill_freq, resolution_freq))
br_plt +  theme(axis.text=element_text(size=18),
                axis.title=element_text(size=18,face="bold")) +
          ylim(0, max(bills_and_resolutions$bill_freq)) + 
          geom_point() + 
          xlab("bills") + 
          ylab("resolutions") +
          geom_abline(intercept=0, slope=1) + 
          geom_text(data=subset(bills_and_resolutions, bill_freq > 5000),
                    aes(bill_freq, resolution_freq, label=toupper(state)),
                    vjust=-.5, size=8) + 
          geom_text(data=subset(bills_and_resolutions, bill_freq < resolution_freq & bill_freq > 100),
                    aes(bill_freq, resolution_freq, label=toupper(state)),
                    vjust=-.5, size=8) 
  

# how many bills Sunlight scraped from each government after the second
# year it started scraping that government
bills_by_state_year <-
    dbGetQuery(con, "SELECT UPPER(c.state) as state, 
                            EXTRACT(YEAR FROM c.created_at) AS year,
                            COUNT(*) AS freq
                     FROM   bill_metadata AS c,
                            -- find minimum year 
                            (SELECT a.state,
                                    MIN(a.year) AS min_year
                             FROM   (SELECT state,
                                            EXTRACT(YEAR FROM created_at) AS year
                                    FROM    bill_metadata) AS a
                            GROUP BY state) as b
                     WHERE  c.state = b.state AND
                            EXTRACT(YEAR FROM created_at) >= b.min_year 
                     GROUP BY c.state, 
                              EXTRACT(YEAR FROM c.created_at)
                     ORDER BY c.state,
                              EXTRACT(YEAR FROM c.created_at);")

# we're missing data for some states in some years
dbGetQuery(con, "SELECT c.state,
                        c.year - 1 AS missing_year
                 FROM (SELECT *,
  	                          b.year - lag(b.year) OVER w AS gap
                       FROM 	(SELECT a.state,
		 		                              a.year,
		 		                              COUNT(*)
		                           FROM 	(SELECT state, 
		 		                                      EXTRACT(YEAR FROM created_at) AS year
		 		                              FROM 	 bill_metadata) AS a
                                		  GROUP BY a.state,
		 		                                       a.year
		                                  ORDER BY a.state,
		 		                                       a.year) AS b
                        WINDOW w AS (ORDER BY b.state, b.year)) AS c
                  WHERE c.gap > 1;")

missing_values <- data.frame(state = c('MT', 'ND', 'NV', 'TX', 'TX'),
                             year = c(2014, 2014, 2012, 2012, 2014),
                             freq = rep(0,5))
bills_by_state_year <- rbind(bills_by_state_year, missing_values)
bills_by_state_year <- bills_by_state_year[ order(bills_by_state_year$state, bills_by_state_year$year), ]

# New Jersey 2012 is wrong. Subtract 2013 number from total here: http://www.njleg.state.nj.us/bills/BillsByNumber.asp
bills_by_state_year$freq[ bills_by_state_year$state == 'NJ' & bills_by_state_year$year == 2012 ] <- 6808

sy_plt <- ggplot(bills_by_state_year, aes(year, freq, color=state))
sy_plt +  theme(legend.position="none", 
                axis.text=element_text(size=18),
                axis.title=element_text(size=18,face="bold")) +
          geom_line(size=2) + 
          ylab("frequency") +
          geom_text(data=data.frame(state=c('NJ', 'TX', 'NJ', 'NY', 'IL', 'TX'),
                                    year=c(2012, 2013, 2014, 2014, 2015.05, 2015), 
                                    freq=c(6850, 11700, 7500, 13200, 7000, 10000)),
                    aes(x=year, y=freq, label=state),
                    vjust=-.5, size=7)
