"""Markdown (film-impression-museum-exhibition-plan.md) -> Postgres content store.

Schema and rationale: CONTEXT.md, docs/adr/0001-postgres-server-content-store.md.

  uv run sync-content                 # parse the Markdown and sync it into Postgres
  uv run sync-content --dry-run       # parse only, print what would be synced
  uv run get-movie the-godfather      # fetch one movie (by movie_slug or id)
  uv run get-movie the-godfather --exhibit   # ...rendered as the LLM's exhibit content

Connection: $DATABASE_URL (default: the docker compose service in compose.yaml).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

DEFAULT_DSN = "postgresql://vrguide:vrguide@localhost:5432/vrguide"
DEFAULT_MD = "film-impression-museum-exhibition-plan.md"

# " 章节1：电影诞生期 (18951910)"  -- the dash between the years was lost in the source
CHAPTER_RE = re.compile(r"^\s*章节\s*(\d+)\s*[：:]\s*(.+?)\s*[(（]\s*(\d{4})\D*(\d{4})\s*[)）]\s*$")
# " 《火车进站》（L'Arrivée d'un train en gare de La Ciotat 1895）"
# "漫威电影宇宙（Marvel Cinematic Universe, 2008至今）"  (no 《》)
# The heading must be the whole line, so body lines that merely start with 《...》 don't match.
MOVIE_RE = re.compile(r"^\s*《?([^《》（）\n]+?)》?\s*（(.+?)[\s,，]*(\d{4})(?:至今)?\s*）\s*$")
END_RE = re.compile(r"^\s*结语\s*$")


@dataclass
class Movie:
    slug: str
    title_zh: str
    title_orig: str
    year: int
    essay_lines: list[str] = field(default_factory=list)

    @property
    def essay_md(self) -> str:
        return "\n".join(self.essay_lines).strip()


@dataclass
class Chapter:
    era_no: int
    title_zh: str
    year_from: int
    year_to: int
    movies: list[Movie] = field(default_factory=list)


def slugify(s: str) -> str:
    s = s.replace("½", " half")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def parse_markdown(text: str) -> list[Chapter]:
    chapters: list[Chapter] = []
    movie: Movie | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if END_RE.match(line):
            break  # 结语 is the museum epilogue, not a movie
        if m := CHAPTER_RE.match(line):
            chapters.append(Chapter(int(m[1]), m[2], int(m[3]), int(m[4])))
            movie = None
            continue
        if m := MOVIE_RE.match(line):
            if not chapters:
                raise ValueError(f"line {lineno}: movie heading before any chapter heading")
            title_orig = m[2].strip()
            movie = Movie(slugify(title_orig), m[1].strip(), title_orig, int(m[3]))
            chapters[-1].movies.append(movie)
            continue
        if movie is not None:
            movie.essay_lines.append(line.strip())
        elif line.strip():
            print(f"warning: line {lineno}: text outside any movie ignored: {line[:40]!r}", file=sys.stderr)

    slugs = [mv.slug for c in chapters for mv in c.movies]
    dup = {s for s in slugs if slugs.count(s) > 1}
    if dup:
        raise ValueError(f"duplicate movie_slug: {sorted(dup)}")
    empty = [mv.slug for c in chapters for mv in c.movies if not mv.essay_md]
    if empty:
        raise ValueError(f"movies with empty essay: {empty}")
    return chapters


def connect(dsn: str | None = None):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN), row_factory=dict_row)


def sync(conn, chapters: list[Chapter]) -> dict[str, int]:
    """Make the DB mirror the Markdown in one transaction.

    Upserts by era_no / movie_slug so ids stay stable across syncs; rows that
    disappeared from the Markdown are deleted."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(files("vr_guide").joinpath("schema.sql").read_text())
        for c in chapters:
            cur.execute(
                """INSERT INTO chapters (era_no, title_zh, year_from, year_to)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (era_no) DO UPDATE SET title_zh = EXCLUDED.title_zh,
                       year_from = EXCLUDED.year_from, year_to = EXCLUDED.year_to
                   RETURNING id""",
                (c.era_no, c.title_zh, c.year_from, c.year_to),
            )
            chapter_id = cur.fetchone()["id"]
            for mv in c.movies:
                cur.execute(
                    """INSERT INTO movies (movie_slug, chapter_id, title_zh, title_orig, year, essay_md)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (movie_slug) DO UPDATE SET chapter_id = EXCLUDED.chapter_id,
                           title_zh = EXCLUDED.title_zh, title_orig = EXCLUDED.title_orig,
                           year = EXCLUDED.year, essay_md = EXCLUDED.essay_md""",
                    (mv.slug, chapter_id, mv.title_zh, mv.title_orig, mv.year, mv.essay_md),
                )
        slugs = [mv.slug for c in chapters for mv in c.movies]
        cur.execute("DELETE FROM movies WHERE movie_slug <> ALL(%s)", (slugs,))
        deleted = cur.rowcount
        cur.execute("DELETE FROM chapters WHERE era_no <> ALL(%s)", ([c.era_no for c in chapters],))
    return {"chapters": len(chapters), "movies": len(slugs), "deleted_movies": deleted}


def get_movie(conn, key: str | int) -> dict | None:
    """Poster ID (movie_slug or numeric id) -> movie row joined with its chapter."""
    col = "m.id" if isinstance(key, int) or str(key).isdigit() else "m.movie_slug"
    return conn.execute(
        f"""SELECT m.id, m.movie_slug, m.title_zh, m.title_orig, m.year, m.essay_md,
                   c.era_no, c.title_zh AS chapter_title,
                   c.year_from AS chapter_year_from, c.year_to AS chapter_year_to
            FROM movies m JOIN chapters c ON c.id = m.chapter_id
            WHERE {col} = %s""",
        (int(key) if col == "m.id" else key,),
    ).fetchone()


def render_exhibit_markdown(row: dict, include_zh_title: bool = False) -> str:
    """get_movie() row -> the "exhibit content" block given to the LLM.

    Same shape as the prompt in scripts/stream_to_tts.py was written against:
      # 章节1:电影诞生期 (1895-1910)
      ## L'Arrivée d'un train en gare de La Ciotat (1895)
      <essay>
    title_zh is left out by default (2026-10-01 decision, docs/llm_prompt_design.md):
    title_orig + "answer in the visitor's language" already covers title questions."""
    chapter = f"# 章节{row['era_no']}:{row['chapter_title']} ({row['chapter_year_from']}-{row['chapter_year_to']})"
    if include_zh_title:
        heading = f"## {row['title_zh']}({row['title_orig']}, {row['year']})"
    else:
        heading = f"## {row['title_orig']} ({row['year']})"
    return f"{chapter}\n\n{heading}\n\n{row['essay_md']}"


def load_exhibit_markdown(key: str | int, include_zh_title: bool = False, dsn: str | None = None) -> str:
    """Poster ID -> exhibit content for the LLM, straight from Postgres."""
    with connect(dsn) as conn:
        row = get_movie(conn, key)
    if row is None:
        raise KeyError(f"no movie for Poster ID {key!r}")
    return render_exhibit_markdown(row, include_zh_title)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync the exhibition Markdown into Postgres")
    ap.add_argument("--md", default=DEFAULT_MD)
    ap.add_argument("--dsn", default=None, help=f"default: $DATABASE_URL or {DEFAULT_DSN}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chapters = parse_markdown(Path(args.md).read_text(encoding="utf-8"))
    if args.dry_run:
        for c in chapters:
            print(f"era{c.era_no} {c.title_zh} ({c.year_from}-{c.year_to})")
            for mv in c.movies:
                print(f"  {mv.slug:40s} {mv.year} {mv.title_zh} / {mv.title_orig}  [{len(mv.essay_md)} chars]")
        return
    with connect(args.dsn) as conn:
        print(json.dumps(sync(conn, chapters)))


def main_get() -> None:
    ap = argparse.ArgumentParser(description="Fetch one movie by Poster ID (movie_slug or id)")
    ap.add_argument("key")
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--exhibit", action="store_true", help="print the LLM exhibit-content Markdown instead of JSON")
    args = ap.parse_args()
    with connect(args.dsn) as conn:
        row = get_movie(conn, args.key)
    if row is None:
        raise SystemExit(f"not found: {args.key}")
    if args.exhibit:
        print(render_exhibit_markdown(row))
        return
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
