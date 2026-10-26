# Simulating a fruit fly in a computer: where the code is, how it works, how to run it

_Researched 2026-10-19. Everything in the "verified" section was actually run in a cloud container (4 CPU cores, 15 GB RAM)._

## The short version

"The fruit fly in a computer" is really three layers stacked together, and all three are open source:

| Layer | What it is | Code / data |
|---|---|---|
| **Wiring diagram (connectome)** | Every neuron (~139k) and synapse (~50M) of an adult female fly brain, mapped from electron microscopy. FlyWire, *Nature* 2024. | [flywire.ai](https://flywire.ai) (data CC BY-NC 4.0). Male whole-CNS version: [MaleCNS v1.0](https://male-cns.janelia.org/) (CC BY 4.0) |
| **Brain simulation** | The connectome turned into a spiking network of leaky integrate-and-fire neurons. Shiu et al., *Nature* 2024. | [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT) |
| **Body simulation** | A physics model of the fly's body (legs, wings, joints) in the MuJoCo engine. | [NeuroMechFly v2 / FlyGym](https://github.com/NeLy-EPFL/flygym) (EPFL) and [flybody](https://github.com/TuragaLab/flybody) (Janelia + DeepMind, *Nature* 2025) |

The headline you probably heard about is **Eon Systems' "virtual embodied fly" (March 2026)**: they wired the Shiu brain model into the NeuroMechFly body so the simulated fly smells/tastes food, walks to it, grooms when dusty, and eats. Their write-up: [eon.systems/updates/embodied-brain-emulation](https://eon.systems/updates/embodied-brain-emulation). Eon did not publish the closed-loop code with that post, but they publish a multi-backend (CPU/GPU) version of the brain model at [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) (GPL-2.0), and community projects have rebuilt the brain+body loop (e.g. [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain)).

**Can we recreate it ourselves? Yes.** The brain model runs on an ordinary laptop/CPU; I ran it here in about 20 seconds per batch of 4 simulated seconds. Building the full embodied loop is a weekend-to-weeks project on top of existing code, not a research lab effort.

## How the brain network works

Source: `model.py` in the Shiu repo (about 300 lines; Brian2 simulator).

1. **Neurons.** Each of the 138,639 neurons is one point (no dendrite shape). Its voltage `v` leaks back to rest (-52 mV) with a 20 ms time constant. When `v` crosses -45 mV it emits a spike, resets, and is silent for 2.2 ms (refractory).
   ```
   dv/dt = (v_0 - v + g) / t_mbr
   dg/dt = -g / tau            (tau = 5 ms synaptic decay)
   ```
2. **Synapses.** Taken straight from the connectome file `Connectivity_783.parquet` (15.1M neuron-to-neuron connections). The weight of a connection is
   `number of synapses between the two neurons × sign × 0.275 mV`.
   The sign (+1 excitatory / -1 inhibitory) comes from the predicted neurotransmitter of the presynaptic neuron (acetylcholine = excitatory; GABA and glutamate = inhibitory). Each spike reaches the target after a 1.8 ms delay.
3. **One free parameter.** Apart from the synapse weight scale (0.275 mV), all constants come from the literature. Nothing is trained. The behavior falls out of the wiring.
4. **Experiments = "optogenetics in silico".** You pick neurons by FlyWire ID and either **activate** them (Poisson spike input, e.g. 150 Hz) or **silence** them (zero their synapses). The model returns the spike times of every neuron in the brain.
5. **Why it's interesting.** The paper showed that activating sugar-taste neurons drives MN9, the motor neuron that extends the proboscis (feeding), and bitter neurons suppress it, and that these predictions held up in real flies more than 90% of the time.

### Files in the repo
- `Completeness_783.csv`: list of neuron IDs (138,639 rows); row index = simulator index.
- `Connectivity_783.parquet` (97 MB): columns `Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index, Postsynaptic_Index, Connectivity, Excitatory, Excitatory x Connectivity`.
- `2023_03_23_*_630_final.*`: the older v630 snapshot the paper used (the code defaults to it).
- `example.ipynb`: tutorial; `figures.ipynb`: reproduces the paper's figures.

## Verified: I ran it

```bash
git clone https://github.com/philshiu/Drosophila_brain_model.git
pip install brian2 pandas pyarrow joblib
```
Then activated the 21 right-side sugar neurons from the tutorial at 150 Hz, 4 trials × 1 s:

- Wall time: **20 s** on 4 cores (plus a one-off ~1.5 min compile on first run).
- 55,974 spikes; **401 neurons** became active out of 138k.
- **MN9 (proboscis motor neuron, ID 720575940660219265) fired at ~87 Hz.** That is the paper's headline "taste sugar → extend proboscis" result, reproduced.

The script I used is [sugar_test.py](sugar_test.py) (the tutorial with `n_run=4`). Run it as `python sugar_test.py <path-to-repo> <output-dir>`.

## How to recreate it ourselves (levels)

**Level 1: run the brain (hours).** Any modern laptop, Python 3.10+, `pip install brian2`. Or click the Colab badge in the repo README. Pick neurons to poke using the FlyWire Codex browser ([codex.flywire.ai](https://codex.flywire.ai)) to find cell types and IDs.

**Level 2: run it faster / bigger (a day).** Use [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain), which re-implements the same model in PyTorch, Brian2CUDA, GeNN and NEST GPU. Needs an NVIDIA GPU with CUDA 12 for the GPU backends (tested on an RTX 4070). Alternatively, writing our own version is easy: the whole network is a sparse 138k × 138k matrix plus the two equations above, about 100 lines of PyTorch/NumPy.

**Level 3: give it a body (days to weeks).** `pip install flygym` (NeuroMechFly v2, MuJoCo). The Eon recipe:
- Every ~15 ms, convert body events to brain input: leg/proboscis contact with food → activate sugar or bitter taste neurons; "dust" on antennae → antennal mechanosensory neurons; camera images → visual neurons via [flyvis](https://github.com/TuragaLab/flyvis).
- Run the brain for that step, read the firing rate of a few descending neurons, and map them to body commands: DNa01/DNa02 → steering, oDN1 → walk forward, MN9 → proboscis, antennal DNs → grooming.
- The walking itself uses FlyGym's built-in leg controllers; the brain only sends the low-dimensional "go / turn / eat / groom" commands.

**Level 4: train it (research).** FlyGM ([arXiv 2602.17997](https://arxiv.org/abs/2602.17997)) uses the connectome as a graph network trained with RL to control walking and flight in flybody. Lots of hobby projects do similar things (driving cars, flying drones); see [awesome-fruit-fly](https://github.com/townie/awesome-fruit-fly).

## Caveats worth knowing
- It is a wiring diagram with very simple neurons. No dendritic computation, no neuromodulators/hormones, no learning, and transmitter signs are predicted, not measured.
- The brain covers the central brain only; the ventral nerve cord (the fly's "spinal cord") is separate data ([MANC](https://doi.org/10.7554/eLife.97769), [BANC](https://github.com/htem/BANC-project)).
- In the Eon demo, many sensor-to-neuron and neuron-to-motor mappings were hand-picked, vision was "decorative", and walking came from pre-trained controllers. Eon say themselves it doesn't prove the connectome alone produces the fly's behavior.
- FlyWire data is non-commercial (CC BY-NC); MaleCNS is CC BY.

## Sources
- Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", *Nature* 2024; code: https://github.com/philshiu/Drosophila_brain_model
- FlyWire connectome: https://flywire.ai, https://codex.flywire.ai
- Eon embodied fly (2026-03-10): https://eon.systems/updates/embodied-brain-emulation ; code: https://github.com/eonsystemspbc/fly-brain
- NeuroMechFly v2 / FlyGym: https://github.com/NeLy-EPFL/flygym
- flybody, "Whole-body physics simulation of fruit fly locomotion", *Nature* 2025: https://www.nature.com/articles/s41586-025-09029-4 ; https://github.com/TuragaLab/flybody
- flyvis (Lappalainen et al., *Nature* 2024): https://github.com/TuragaLab/flyvis
- FlyGM: https://arxiv.org/html/2602.17997v3
- Project list: https://github.com/townie/awesome-fruit-fly
- Berkeley news: https://vcresearch.berkeley.edu/news/researchers-simulate-fly-brain-laptop-human-brain-next
