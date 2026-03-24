#!/usr/bin/env bash
# Record a terminal demo of deltatau-audit for README/social media.
#
# Option 1: Use asciinema (recommended)
#   pip install asciinema
#   asciinema rec demo.cast -c "bash scripts/record_demo.sh --run"
#   # Convert to GIF: agg demo.cast demo.gif
#
# Option 2: Run directly and screenshot the output
#   bash scripts/record_demo.sh --run

set -e

if [ "$1" = "--run" ]; then
    echo ""
    echo "$ pip install deltatau-audit"
    echo "Successfully installed deltatau-audit-1.0.0"
    echo ""
    sleep 1

    echo "$ deltatau-audit demo --episodes 10"
    echo ""
    python -m deltatau_audit demo --episodes 10 --out /tmp/demo_recording 2>&1

    echo ""
    echo "# Open the HTML report:"
    echo "$ open /tmp/demo_recording/baseline/index.html"
    echo ""
else
    echo "Usage:"
    echo "  Record with asciinema:"
    echo "    asciinema rec demo.cast -c 'bash scripts/record_demo.sh --run'"
    echo ""
    echo "  Run directly:"
    echo "    bash scripts/record_demo.sh --run"
    echo ""
    echo "  Convert to GIF (requires agg):"
    echo "    agg demo.cast demo.gif --cols 100 --rows 30"
fi
