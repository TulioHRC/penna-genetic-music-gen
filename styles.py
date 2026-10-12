"""
Presets de estilo do arranjo: timbres (programas General MIDI), kit de bateria e como cada
pista se comporta. O *ritmo* da bateria vem preferencialmente do groove extraído do áudio de
referência (groove_extractor.py); o `fallback_groove` só é usado quando não há áudio.
"""

from dataclasses import dataclass
from typing import Optional

from groove_extractor import Groove

# GM Drum Map
KICK, SNARE, HAND_CLAP, CLOSED_HAT = 36, 38, 39, 42


@dataclass(frozen=True)
class StylePreset:
    name: str
    lead_program: int
    bass_program: int
    harmony_program: Optional[int]  # None: estilo sem acordes sustentados
    drum_kit: int                   # preset do banco de bateria (0 = Standard, 25 = 808/909)
    snare_pitch: int                # caixa (rock) ou palma (funk)
    bass_center_pitch: int          # região do baixo
    bass_fifth_on_offbeats: bool    # rock: alterna fundamental/quinta; funk: 808 só na fundamental
    bass_legato: float              # fração do tempo até o próximo bumbo que a nota do baixo sustenta
    phrase_fills: bool              # prato no início e virada no fim de cada frase de 4 compassos
    fallback_groove: Groove


STYLES = {
    "rock": StylePreset(
        name="rock",
        lead_program=29,          # Overdriven Guitar
        bass_program=33,          # Electric Bass (finger)
        harmony_program=48,       # String Ensemble
        drum_kit=0,               # Standard
        snare_pitch=SNARE,
        bass_center_pitch=40,     # E2
        bass_fifth_on_offbeats=True,
        bass_legato=0.8,
        phrase_fills=True,
        fallback_groove=Groove.from_steps(120.0, kick=(0, 6, 8), snare=(4, 12), hat=range(0, 16, 2)),
    ),
    "funk": StylePreset(
        name="funk",
        lead_program=81,          # Saw Lead
        bass_program=38,          # Synth Bass 1 (papel do 808)
        harmony_program=None,     # funk carioca é seco: voz/lead + batida + grave
        drum_kit=25,              # 808/909
        snare_pitch=HAND_CLAP,
        bass_center_pitch=33,     # A1: grave de 808
        bass_fifth_on_offbeats=False,
        bass_legato=0.95,
        phrase_fills=False,
        # Tamborzão (3+3+2): usado só se não houver áudio de referência para extrair o groove
        fallback_groove=Groove.from_steps(130.0, kick=(0, 3, 6, 10, 14), snare=(4, 12), hat=(2, 7, 10, 15)),
    ),
}
