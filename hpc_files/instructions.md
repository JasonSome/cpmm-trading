# Running the Code on Ginsburg

## Logging In

Open a terminal and type:

```bash
ssh <user>@burg.rcs.columbia.edu
```

When prompted, enter your password and press **Enter**.

---

## Navigation

When you log in, you'll be placed on a **login node**. Do **not** run any code directly here.

Useful terminal commands:

- `ls` — List contents of the current directory  
- `cd <directory>` — Change to a specified directory  
- `cd ..` — Move up one directory level  
- `cat <filename>` — Display the contents of a file  
- `mkdir <foldername>` — Create a new directory  
- `rm <file>` — Delete a file or directory  
- `exit` — End your session on Ginsburg

---

## Text Editing

Use `nano` to edit or create text files:

```bash
nano <filename>
```

- Use arrow keys to navigate.
- To save: press `Ctrl + O`, then `Enter`
- To exit: press `Ctrl + X`
- Additional shortcuts are displayed at the bottom of the nano window.

---

## Moving Files to the HPC Cluster

Run the following command **outside** your Ginsburg session (from your local terminal). Use `motion` for transfers:

```bash
scp <local_file_path> <user>@motion.rcs.columbia.edu:<remote_directory>
```

**Example:** Move a local Python file to the `/AMM_sims` directory on Ginsburg:

```bash
scp fast_AMM_sim_functions.py <user>@motion.rcs.columbia.edu:~/AMM_sims/
```

> 🔐 You will be prompted to enter your password to complete the transfer.

---

## Moving Files from Ginsburg to Your Local Machine

Also done **outside** of the Ginsburg session:

```bash
scp <user>@motion.rcs.columbia.edu:~/<remote_path>/<filename> <local_directory>
```

**Example:** Download a `.pkl` file from `/AMM_sims` on Ginsburg to your local `~/Downloads`:

```bash
scp <user>@motion.rcs.columbia.edu:~/AMM_sims/all_outputs_eta0_0.003_mu_0.0_buy_2500_sell_2500.pkl ~/Downloads
```

> 🔐 You will be prompted to enter your password to complete the transfer.

---

## Running the Simulation Code

Ensure the following two files are in your working directory on Ginsburg:

- `sim.sh` – Shell script for submitting the job  
- `run_AMM_sim.py` – Python simulation code

### Step 1: Edit the Simulation Parameters

```bash
nano sim.sh
```

Make the necessary edits to match your simulation specifications.

### Step 2: Submit the Job

```bash
sbatch sim.sh
```

### Step 3: Monitor Job Status

To check the status of your jobs:

```bash
squeue -u $USER
```
