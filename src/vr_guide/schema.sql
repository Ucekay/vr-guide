-- Content store schema (CONTEXT.md / ADR 0001). Idempotent.
CREATE TABLE IF NOT EXISTS chapters (
    id        serial PRIMARY KEY,
    era_no    integer NOT NULL UNIQUE,
    title_zh  text    NOT NULL,
    year_from integer,
    year_to   integer
);

CREATE TABLE IF NOT EXISTS movies (
    id         serial PRIMARY KEY,
    movie_slug text    NOT NULL UNIQUE,
    chapter_id integer NOT NULL REFERENCES chapters(id),
    title_zh   text    NOT NULL,
    title_orig text    NOT NULL,
    year       integer,
    essay_md   text    NOT NULL
);
