import psycopg2
from tabulate import tabulate


def fetch_all(cursor):
    colnames = [desc[0] for desc in cursor.description]
    records = cursor.fetchall()
    return [{colname: value for colname, value in zip(colnames, record)} for record in records]


conn = psycopg2.connect(host='localhost', port='5432', dbname='odscourse', user='postgres', password='secret')
cursor = conn.cursor()

# 1. How old were the youngest male and female participants of the 1996 Olympics? (12, 14)
cursor.execute("""
    SELECT MIN(age) FROM athlete_events
    WHERE age!= 0 AND sex = 'F' AND year = 1996
""")
print(tabulate(fetch_all(cursor), "keys", "psql"))

cursor.execute("""
    SELECT MIN(age) FROM athlete_events
    WHERE age!= 0 AND sex = 'M' AND year = 1996
""")
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 2. What was the percentage of male gymnasts among all the male participants of the 2000 Olympics? (0,015)
# Round the answer to the first decimal.

cursor.execute(
    """
    SELECT ROUND((COUNT(DISTINCT(athlete_id)) / (SELECT COUNT(DISTINCT(athlete_id))
    FROM athlete_events WHERE sex = 'M' AND year = 2000)::numeric), 3)
        FROM athlete_events
        WHERE sex = 'M' AND year = 2000 AND sport = 'Gymnastics'
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 3. What are the mean and standard deviation of height for female basketball players participated in the 2000 Olympics?
#(182.4, 9.1)

cursor.execute(
    """
    SELECT ROUND((AVG(con.height)::numeric), 1) AS "average_height",
    ROUND((STDDEV(con.height)::numeric), 1) AS "stddevastation_height"
    FROM (SELECT DISTINCT athlete_id, height FROM athlete_events
    WHERE sex = 'F' AND year = 2000 AND sport = 'Basketball') AS con
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 4. Find a sportsperson who participated in the 2002 Olympics,
# with the highest weight among other participants of the same Olympics. What sport did he or she do?
#Bobsleigh

cursor.execute(
    """
    SELECT sport
    FROM athlete_events
    WHERE weight=(SELECT MAX(weight) FROM athlete_events WHERE year = 2002) AND year = 2002
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 5. How many times did Pawe Abratkiewicz participate in the Olympics held in different years? #3
cursor.execute(
    """
    SELECT COUNT(DISTINCT(athlete_events.games)) AS "amount of games"
    FROM athlete_events
    WHERE name = 'Pawe Abratkiewicz'
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 6. How many silver medals in tennis did Australia win at the 2000 Olympics? #2
cursor.execute(
    """
     SELECT COUNT(medal) FROM athlete_events
     WHERE Medal = 'Silver' and Sport = 'Tennis' and Year = 2000 and Team = 'Australia';
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 7. Is it true that Switzerland won fewer medals than Serbia at the 2016 Olympics? #Yes
cursor.execute(
    """
    SELECT team, COUNT(medal) FROM athlete_events
    WHERE medal != 'NA' AND (team = 'Serbia' OR team = 'Switzerland') AND year = 2016
    GROUP BY team
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 9. Is it true that there were Summer Olympics held in Lake Placid?
# Is it true that there were Winter Olympics held in Sankt Moritz?
#Only in Sankt Moritz

cursor.execute(
    """
    SELECT city, COUNT(DISTINCT games) FROM athlete_events
    WHERE (city = 'Lake Placid' AND season = 'Summer') OR (city = 'Sankt Moritz' AND season = 'Winter')
    GROUP BY city
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 10. What is the absolute difference between the number of unique sports at the 1996 Olympics and 2016 Olympics?
#3
cursor.execute(
    """
    SELECT DISTINCT ABS(
        (SELECT COUNT (DISTINCT sport) FROM athlete_events WHERE year = 1996) - 
        (SELECT COUNT (DISTINCT sport) FROM athlete_events WHERE year = 2016)
    )::int AS diff
    FROM athlete_events
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))

# 8. What age category did the fewest and the most participants of the 2014 Olympics belong to?
#most - [25-35], fewest - [45-55]
cursor.execute(
    """
    SELECT
        CASE
            WHEN 15 <= age AND 25 > age THEN 15
            WHEN 25 <= age AND 35 > age THEN 25
            WHEN 35 <= age AND 45 > age THEN 35
            WHEN 45 <= age AND 55 >= age THEN 45
        END AS age_group,
    COUNT (DISTINCT athlete_id)
    FROM athlete_events
    WHERE year = 2014
    GROUP BY age_group
    """
)
print(tabulate(fetch_all(cursor), "keys", "psql"))
