# Fly brain on GPU + a brain that grows

Two scripts, both plain PyTorch, both run on CPU or an NVIDIA GPU (`--device cuda`).

## 1. `flybrain_torch.py`: the fly brain, GPU-ready

A re-implementation of the Shiu et al. model (`model.py` in `Drosophila_brain_model`) in about 200 lines.

- **Same model.** Same equations and constants. Checked against the original Brian2 code with identical inputs: over 200 ms of simulated time both produced 2,565 spikes and every spike matched except one, which came one time step (0.1 ms) later (float rounding). Over 30 random trials the firing rates of all neurons correlate at 0.9997, and MN9 fires at 84.5 Hz vs 83.8 Hz in Brian2.
- **Two Brian2 details I had to copy** to get that match: a neuron is frozen for 21 steps (not 22) after a spike, and synaptic input that arrives while a neuron is refractory is thrown away.
- **Why it's faster.** All trials run as one batch, and each step only touches the outgoing synapses of neurons that actually fired (about 400 of 139k are active), so cost tracks activity, not the 15M connections.
- **Speed on CPU** (4 cores, this container): 30 trials × 1 s in about 4 min, about the same as Brian2's 7 min. The GPU is where it should pay off; we haven't measured that yet.

```
python flybrain_torch.py --repo Drosophila_brain_model --version 783 --trials 30 --device cuda
```
Output: `flybrain_rates.csv` (firing rate of every active neuron). The `FlyBrain` class also has `silence()`, `record=` voltage traces, and `input_hits=` for feeding in your own sensory spikes (that's the hook for connecting a body later).

## 2. `growing_brain.py`: trainable spiking network that grows

The "let a program build the brain and train it" part, small enough to understand fully.

- **Neurons.** Same leaky integrate-and-fire style as the fly (voltage, synaptic current, threshold, reset).
- **Synapses.** A recurrent weight matrix plus a connectivity mask, so most neuron pairs are not connected (10% start connected).
- **Training.** Backprop through time on GPU. Spikes are not differentiable, so the backward pass uses a smooth stand-in (a "surrogate gradient"), which is the standard trick for training spiking networks.
- **Growing.** When accuracy stalls, `grow()` adds 50% more neurons with new random synapses. Old weights are kept, new neurons start with zero output weight, and the optimizer's memory is carried over, so the network computes exactly the same thing right after growing (measured output change: 0.0) and training simply continues.
- **Rewiring.** Every 100 steps the weakest 2% of synapses are pruned and the same number of new random ones sprout (they start at zero and are learned).
- **Curriculum.** The task is recognising noisy, time-jittered spike patterns. Each time the brain hits 90% accuracy, the number of classes doubles, which adds output neurons the same function-preserving way.
- **Continue later.** The brain is saved to `brain.pt`; `--resume brain.pt` keeps training (and growing) from there.

Run in this container (CPU): it started at 16 neurons and 4 classes. Over 5,500 steps (about 2.5 min) it grew on its own four times, to 81 neurons and 2,245 synapses, and reached 64 classes at 85% accuracy.

```
python growing_brain.py --device cuda --steps 5000
python growing_brain.py --device cuda --resume brain.pt --steps 5000
```

## What this is and isn't
- It is a working skeleton for "build, train, grow, continue" with spiking neurons. Each piece (neuron model, growth rule, rewiring rule, task) can be swapped.
- The weight matrix is dense with a mask, which is simple and fast on GPU up to roughly 10-30k neurons (it needs N² memory). Fly scale (139k neurons) needs a sparse matrix like the one in `flybrain_torch.py`. That is the next engineering step if we want to grow that big.
- It does not yet respect biology such as fixed excitatory/inhibitory neuron types (Dale's law), local learning rules instead of backprop, or growth that follows where activity is. These are the obvious research knobs.

## Ideas for next steps
1. Benchmark both scripts on your GPU.
2. Seed the growing brain with a real piece of the fly connectome (for example the taste-to-feeding circuit) and let it grow from there.
3. Make growth smarter: add neurons where error or activity is highest instead of at random.
4. Give it a body: FlyGym (`pip install flygym`) for the fly, or a simple game environment, using `input_hits` / outputs as the sensor and motor interface.
