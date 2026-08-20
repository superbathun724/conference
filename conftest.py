"""pytest가 어디서 실행되든 airgap 패키지를 찾을 수 있도록 루트를 sys.path에 넣는다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
