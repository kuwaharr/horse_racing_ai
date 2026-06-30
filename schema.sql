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

CREATE INDEX IF NOT EXISTS idx_runner_horse_id
    ON runner(horse_id);

CREATE TABLE IF NOT EXISTS horse (
    horse_id TEXT PRIMARY KEY,
    horse_name TEXT,
    sire_id TEXT,
    sire_name TEXT,
    dam_id TEXT,
    dam_name TEXT,
    broodmare_sire_id TEXT,
    broodmare_sire_name TEXT,
    pedigree_fetched_at TEXT,
    pedigree_fetch_status TEXT NOT NULL DEFAULT 'pending',
    pedigree_fetch_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (pedigree_fetch_status IN ('pending', 'fetched', 'failed', 'not_found'))
);

CREATE INDEX IF NOT EXISTS idx_horse_pedigree_fetch_status
    ON horse(pedigree_fetch_status);

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

CREATE TABLE IF NOT EXISTS pre_race_odds_snapshot (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,
    bet_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'netkeiba',
    snapshot_at TEXT NOT NULL,
    race_date TEXT,
    post_time TEXT,
    post_time_at TEXT,
    minutes_to_post REAL,
    time_bucket TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    raw_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (bet_type IN ('win', 'place', 'wide', 'trio')),
    CHECK (status IN ('fetched', 'no_odds', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_pre_race_odds_snapshot_race_bucket
    ON pre_race_odds_snapshot(race_id, time_bucket, bet_type);

CREATE INDEX IF NOT EXISTS idx_pre_race_odds_snapshot_post_time
    ON pre_race_odds_snapshot(post_time_at);

CREATE TABLE IF NOT EXISTS pre_race_win_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number INTEGER NOT NULL,
    odds REAL,
    CHECK (odds IS NULL OR odds > 0),
    PRIMARY KEY (snapshot_id, horse_number),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pre_race_place_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number INTEGER NOT NULL,
    odds_min REAL,
    odds_max REAL,
    CHECK (odds_min IS NULL OR odds_min > 0),
    CHECK (odds_max IS NULL OR odds_max > 0),
    CHECK (odds_min IS NULL OR odds_max IS NULL OR odds_min <= odds_max),
    PRIMARY KEY (snapshot_id, horse_number),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pre_race_wide_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number_1 INTEGER NOT NULL,
    horse_number_2 INTEGER NOT NULL,
    odds_min REAL,
    odds_max REAL,
    CHECK (horse_number_1 < horse_number_2),
    CHECK (odds_min IS NULL OR odds_min > 0),
    CHECK (odds_max IS NULL OR odds_max > 0),
    CHECK (odds_min IS NULL OR odds_max IS NULL OR odds_min <= odds_max),
    PRIMARY KEY (snapshot_id, horse_number_1, horse_number_2),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pre_race_trio_odds (
    snapshot_id INTEGER NOT NULL,
    horse_number_1 INTEGER NOT NULL,
    horse_number_2 INTEGER NOT NULL,
    horse_number_3 INTEGER NOT NULL,
    odds REAL,
    CHECK (horse_number_1 < horse_number_2 AND horse_number_2 < horse_number_3),
    CHECK (odds IS NULL OR odds > 0),
    PRIMARY KEY (snapshot_id, horse_number_1, horse_number_2, horse_number_3),
    FOREIGN KEY (snapshot_id) REFERENCES pre_race_odds_snapshot(snapshot_id) ON DELETE CASCADE
);
