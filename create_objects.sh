#!/bin/bash
echo "creating pedestrians and vehicles..."
#python3 spawn_npc.py & python3 main.py &
python3 spawn_npc.py & sleep 5 & python3 client_bounding_boxes.py &
