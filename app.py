from flask import Flask, request, jsonify, abort
from db import db
from src.similar_maps import get_similar_maps, build_json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route("/api/similar")
def similar():
    beatmap_id = request.args.get("beatmap_id", type=int)
    if not beatmap_id:
        abort(400, "beatmap_id query-param required, e.g. /similar?beatmap_id=2233275&mods=0")
    
    mods = request.args.get("mods", type=int, default=0)

    # TODO: Add mods into similar maps algorithm
    beatmaps = get_similar_maps(beatmap_id, mods, max_maps=50)

    if not beatmaps:
        return jsonify({"similar": beatmaps})
    
    attributes = build_json(beatmaps)
    
    """
    Replacing this database stuff with osu!api for now.
    Could reuse this code as a filtering option later, but
    the osu!api seems like a better solution for now since
    I'm only making one call per search and it gives me more
    information.
    """
    # ids_query = ", ".join(["%s"] * len(ids))
    # sql_query = f"""
    #     SELECT beatmap_id, user_id, filename, version,
    #            total_length, hit_length,
    #            diff_drain, diff_size, diff_overall, diff_approach,
    #            last_update, difficultyrating, playcount, bpm
    #     FROM osu_beatmaps
    #     WHERE beatmap_id IN ({ids_query})
    #     ORDER BY FIELD(beatmap_id, {ids_query})
    # """

    # with db() as cur:
    #     cur.execute(sql_query, ids + ids)
    #     rows = cur.fetchall()

    # for row, dist in zip(rows, distances):
    #     row["distance"] = dist

    return jsonify({"similar": attributes})
