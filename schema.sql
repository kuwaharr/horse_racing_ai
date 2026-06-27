PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS race (
    race_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    race_number INTEGER NOT NULL,
    post_time_min INTEGER,
    surface_id INTEGER,
    distance INTEGER,
    course_direction_id INTEGER,
    course_layout_id INTEGER,
    course_variant_id INTEGER,
    weather_id INTEGER,
    track_condition_id INTEGER,
    race_size INTEGER,
    UNIQUE (date, track_id, race_number),
    CHECK (length(date) = 10)
);

CREATE TABLE IF NOT EXISTS runner (
    race_id TEXT NOT NULL,
    gate INTEGER,
    horse_number INTEGER NOT NULL,
    finish INTEGER,
    status_id INTEGER NOT NULL,
    horse_name TEXT NOT NULL,
    horse_id TEXT,
    sex_id INTEGER,
    age INTEGER,
    jockey_raw TEXT,
    jockey_id TEXT,
    weight REAL,
    time_sec REAL,
    finish_diff REAL,
    popularity INTEGER,
    finish_3f REAL,
    corner_1 INTEGER,
    corner_2 INTEGER,
    corner_3 INTEGER,
    corner_4 INTEGER,
    corner_count INTEGER,
    stable_id INTEGER,
    trainer_raw TEXT,
    trainer_id TEXT,
    horse_weight INTEGER,
    horse_weight_diff INTEGER,
    PRIMARY KEY (race_id, horse_number),
    FOREIGN KEY (race_id) REFERENCES race(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS place_odds (
    race_id TEXT NOT NULL,
    horse_number INTEGER NOT NULL,
    odds_min REAL,
    odds_max REAL,
    CHECK (odds_min IS NULL OR odds_min > 0),
    CHECK (odds_max IS NULL OR odds_max > 0),
    CHECK (odds_min IS NULL OR odds_max IS NULL OR odds_min <= odds_max),
    PRIMARY KEY (race_id, horse_number),
    FOREIGN KEY (race_id, horse_number) REFERENCES runner(race_id, horse_number) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS win_odds (
    race_id TEXT NOT NULL,
    horse_number INTEGER NOT NULL,
    odds REAL,
    CHECK (odds IS NULL OR odds > 0),
    PRIMARY KEY (race_id, horse_number),
    FOREIGN KEY (race_id, horse_number) REFERENCES runner(race_id, horse_number) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS wide_odds (
    race_id TEXT NOT NULL,
    horse_number_1 INTEGER NOT NULL,
    horse_number_2 INTEGER NOT NULL,
    odds_min REAL,
    odds_max REAL,
    CHECK (horse_number_1 < horse_number_2),
    CHECK (odds_min IS NULL OR odds_min > 0),
    CHECK (odds_max IS NULL OR odds_max > 0),
    CHECK (odds_min IS NULL OR odds_max IS NULL OR odds_min <= odds_max),
    PRIMARY KEY (race_id, horse_number_1, horse_number_2),
    FOREIGN KEY (race_id, horse_number_1) REFERENCES runner(race_id, horse_number) ON DELETE CASCADE,
    FOREIGN KEY (race_id, horse_number_2) REFERENCES runner(race_id, horse_number)
);

CREATE TABLE IF NOT EXISTS trio_odds (
    race_id TEXT NOT NULL,
    horse_number_1 INTEGER NOT NULL,
    horse_number_2 INTEGER NOT NULL,
    horse_number_3 INTEGER NOT NULL,
    odds REAL,
    CHECK (horse_number_1 < horse_number_2 AND horse_number_2 < horse_number_3),
    CHECK (odds IS NULL OR odds > 0),
    PRIMARY KEY (race_id, horse_number_1, horse_number_2, horse_number_3),
    FOREIGN KEY (race_id, horse_number_1) REFERENCES runner(race_id, horse_number) ON DELETE CASCADE,
    FOREIGN KEY (race_id, horse_number_2) REFERENCES runner(race_id, horse_number),
    FOREIGN KEY (race_id, horse_number_3) REFERENCES runner(race_id, horse_number)
);
