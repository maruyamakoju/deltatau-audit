"""
Meta-Dashboard Generator: Aggregating multiple audit reports.
"""

import datetime
import json
import os


def generate_meta_dashboard(root_dir: str, output_path: str):
    reports = []
    for d in os.listdir(root_dir):
        summary_path = os.path.join(root_dir, d, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
                reports.append(
                    {
                        "name": d.capitalize(),
                        "score": data["summary"]["deployment_score"],
                        "rating": data["summary"]["deployment_rating"],
                        "quadrant": data["summary"]["quadrant"],
                        "path": os.path.join(d, "index.html"),
                    }
                )

    html = f"""
    <html>
    <head>
        <title>Meta-Audit Dashboard</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; background: #f4f4f9; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
            .score {{ font-size: 24px; font-weight: bold; }}
            .PASS {{ color: green; }} .DEGRADED {{ color: orange; }} .FAIL {{ color: red; }}
            a {{ text-decoration: none; color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>🔍 Meta-Audit Research Dashboard</h1>
        <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <div class="grid">
    """

    for r in reports:
        html += f"""
        <div class="card">
            <h3>{r["name"]}</h3>
            <div class="score {r["rating"]}">{r["score"]:.2f} ({r["rating"]})</div>
            <p>Quadrant: <code>{r["quadrant"]}</code></p>
            <a href="{r["path"]}">View Full Report →</a>
        </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        generate_meta_dashboard(sys.argv[1], os.path.join(sys.argv[1], "dashboard.html"))
