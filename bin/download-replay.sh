#!/usr/bin/env bash
rm -rf ./replays/${1}
scp -r kbrendan@10.38.1.194:~/StarCraftII/Replays/saves/${1}/replays/ ./replays/${1}
