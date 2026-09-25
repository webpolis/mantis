#!/usr/bin/env python3
"""
Generate MANTIS architecture diagram for scientific publication.
Creates a TikZ LaTeX file that can be compiled to PDF.
"""

import os

tikz_content = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, calc, backgrounds, shadows}

% Color Palette (Nature/Science style)
\definecolor{mantisBlue}{RGB}{66, 133, 244}   % Main Process
\definecolor{mantisRed}{RGB}{234, 67, 53}     % Controller/Decision
\definecolor{mantisGreen}{RGB}{52, 168, 83}   % Memory
\definecolor{mantisYellow}{RGB}{251, 188, 5}  % Warning/Critic
\definecolor{mantisGray}{RGB}{240, 240, 240}  % Backgrounds
\definecolor{darkGray}{RGB}{100, 100, 100}

\begin{document}

\begin{tikzpicture}[
    node distance=2cm and 2.5cm,
    font=\sffamily\small,
    >={Latex[width=2mm,length=2mm]},
    % Node Styles
    process/.style={
        rectangle,
        rounded corners,
        minimum width=2.5cm,
        minimum height=1cm,
        text centered,
        draw=mantisBlue!80,
        fill=mantisBlue!10,
        thick
    },
    controller/.style={
        rectangle,
        rounded corners=5pt,
        minimum width=3cm,
        minimum height=1.5cm,
        text centered,
        draw=mantisRed!80,
        fill=mantisRed!10,
        thick,
        drop shadow
    },
    memory/.style={
        cylinder,
        shape border rotate=90,
        aspect=0.25,
        minimum width=2cm,
        minimum height=1.5cm,
        text centered,
        draw=mantisGreen!80,
        fill=mantisGreen!10,
        thick
    },
    moe_model/.style={
        rectangle,
        rounded corners,
        minimum width=3.5cm,
        minimum height=2cm,
        text centered,
        draw=mantisBlue!80,
        fill=mantisBlue!20,
        thick,
        double
    },
    decision/.style={
        diamond,
        aspect=1.5,
        minimum width=1.8cm,
        minimum height=1cm,
        draw=mantisRed!80,
        fill=white,
        font=\small
    },
    critic/.style={
        trapezium,
        trapezium left angle=70,
        trapezium right angle=110,
        minimum width=2.5cm,
        minimum height=1cm,
        draw=mantisYellow!100!orange,
        fill=mantisYellow!20,
        thick
    },
    % Connector Styles
    flow/.style={->, thick, darkGray},
    control/.style={->, thick, mantisRed, dashed},
    memory_link/.style={->, thick, mantisGreen},
    consolidation/.style={->, double, thick, mantisGreen!60}
]

% --- Step 1: Query Input & Full Encoding ---

\node (input) [process, fill=white] {User Query};

% Base MoE for encoding (FIRST PASS)
\node (encoder) [moe_model, above=1.5cm of input, align=center] {
    \textbf{Base MoE Model}\\
    {\footnotesize (base preset: 24 layers, GQA,}\\
    {\footnotesize 6.7B params, 1.9B active)}\\
    {\tiny One query pass: cache, hidden states, mean pool}
};
\draw[flow] (input) -- (encoder);

% Uncertainty and confidence from the query's next-token distribution
\node (uncertainty) [process, right=1.5cm of encoder, scale=0.7, align=center] {Query\\predictability\\{\tiny (next-token entropy)}};
\draw[flow] (encoder.east) -- (uncertainty.west);

% --- Step 2: Meta-Controller Routing ---

\node (meta) [controller, above=1.5cm of encoder, align=center] {
    \textbf{Meta-Controller}\\
    {\footnotesize (6 Residual MLP Blocks)}
};
\draw[flow] (encoder) -- node[right, font=\tiny] {Query embedding} (meta);

% State Encoder
\node (state) [process, left=2cm of meta, scale=0.7, align=center] {State Encoder\\{\tiny query entropy, top-1 prob,}\\{\tiny context \& memory fill}};
\draw[flow] (uncertainty.south) -- ++(0,-0.9) -| (state.south);
\draw[flow] (state) -- (meta);

% --- Step 3: Routing Decision ---

\node (early_check) [decision, above=1.5cm of meta, align=center] {Bypass?};
\draw[control] (meta.north) -- node[right, font=\tiny] {Gate 1} (early_check);

% --- LEFT PATH: Bypass (decodes from the cached query prefill) ---

\node (simple_gen) [moe_model, left=5cm of early_check, align=center, scale=0.85] {
    \textbf{Base MoE}\\
    {\footnotesize Decode from cached prefill}\\
    {\tiny (full depth; no memory, bias or critic)}
};
\draw[flow] (early_check) -- node[above, font=\tiny, align=center] {Yes: Gate 1 open and\\query entropy $<$ 0.2} (simple_gen);

\node (simple_out) [process, above=1.5cm of simple_gen, fill=white] {Response};
\draw[flow] (simple_gen) -- (simple_out);

% --- RIGHT PATH: Memory + Expert Routing ---

\node (mem_gate) [process, right=4.5cm of early_check, scale=0.8, align=center] {Memory\\Gates};
\draw[control] (early_check) -- node[above, font=\tiny] {No} (mem_gate);

% L2: Episodic Memory
\node (episodic) [memory, above=0.8cm of mem_gate, align=center] {L2: Episodic\\{\tiny (SSM keys, provenance, hits)}};
\draw[control] ($(meta.east)+(0, 0.2)$) to[out=15, in=180] node[pos=0.4, above, font=\tiny] {Gate 2} (episodic.west);
\draw[memory_link] (episodic.south) -- (mem_gate.north);

% L3: Semantic Memory
\node (semantic) [memory, below=0.8cm of mem_gate, align=center] {L3: Semantic\\{\tiny (FAISS, namespaces, trust)}};
\draw[control] ($(meta.east)-(0, 0.2)$) to[out=-15, in=180] node[pos=0.4, below, font=\tiny] {Gate 3} (semantic.west);
\draw[memory_link] (semantic.north) -- (mem_gate.south);

% Consolidation - route around to avoid crossing mem_gate
\coordinate (consol_turn) at ($(semantic.east)+(0.6,0)$);
\draw[consolidation] (episodic.east) to[out=0, in=90] (consol_turn) to[out=-90, in=0] ($(semantic.east)+(0,-0.25)$);
\node[font=\tiny, text=mantisGreen!80!black, align=left, anchor=west] at ($(consol_turn)+(0.1,-0.6)$) {Consolidator:\\evictions and hits,\\records kept whole};

% Memory prompt
\node (context) [process, right=2cm of mem_gate, scale=0.8, align=center] {Evidence Prompt\\{\tiny (reranked, budgeted,}\\{\tiny source ids + query)}};
\draw[flow] (mem_gate) -- (context);

% --- Step 4: Generation with Expert Routing ---

\node (expert_gen) [moe_model, above=2.5cm of context, align=center] {
    \textbf{Base MoE Model}\\
    {\footnotesize Generation; prefill reused}\\
    {\footnotesize when no evidence}
};
\draw[flow] (context) -- (expert_gen);

% Expert bias from meta-controller
\draw[control] ($(meta.north)+(0.6, 0)$) to[out=90, in=180] node[pos=0.5, left=5pt, font=\tiny, align=right] {Gate 4: Expert Bias\\(bounded, per layer,\\off by default)} (expert_gen.west);

% MoE layers (conceptual representation)
\node (moe_layers) [process, above right=0.3cm and 0.8cm of expert_gen, scale=0.6, align=center] {Per-Layer\\MoE\\(8 experts,\\top-2)};

% --- Step 5: Verification (Optional) ---

\node (critic) [critic, right=3cm of expert_gen, align=center] {Critic Model\\{\footnotesize (80M, over base}\\{\footnotesize hidden states)}};
\draw[control] ($(meta.south east)+(-0.3, 0)$) |- ++(0,-0.4) -| node[pos=0.75, right, font=\tiny] {Gate 5} (critic.south);
\draw[control, dashed] (expert_gen.east) -- node[above, font=\tiny] {Verify with evidence} (critic.west);
\draw[control, dashed] (critic.south) to[out=-90, in=0] node[pos=0.5, right, font=\tiny, align=left] {Rejected: retrieve\\once more, regenerate} (context.east);

% --- Final Output ---

\node (output) [process, above=1.5cm of expert_gen, fill=white, double] {Generated Response};
\draw[flow] (expert_gen) -- (output);
\draw[control, dashed] (critic.north) to[out=120, in=0] node[pos=0.6, above right, font=\tiny, align=left] {Abstain if score\\still $<$ 0.6} (output.east);

% Merge early exit path
\draw[flow] (simple_out.north) |- (output.west);

% Every answered interaction is written to episodic memory (never an abstention)
\draw[memory_link, dotted] ($(output.south west)+(0.3,0)$) to[out=-150, in=90] node[pos=0.45, left=3pt, font=\tiny, text=mantisGreen!80!black, align=right] {Store query + response\\with provenance\\(not abstentions)} (episodic.north);

% --- Backgrounds/Grouping ---

\begin{pgfonlayer}{background}
    % Memory System
    \node [fit=(episodic) (semantic) (mem_gate), fill=mantisGreen!5, rounded corners, draw=mantisGreen!20, thick, label={[mantisGreen, font=\bfseries]below:Memory Hierarchy}] {};

    % Meta-Controller decision zone
    \node [fit=(meta) (state) (early_check), fill=mantisRed!3, rounded corners, draw=none] {};
\end{pgfonlayer}

% Legend
\node[font=\tiny, align=left, anchor=north west] at ($(simple_gen.west |- input.north)+(0,0.4)$) {
    \textbf{5 Routing Gates:}\\
    1. Bypass optional components (Bernoulli)\\
    2. Episodic Access (Bernoulli)\\
    3. Semantic Access (Bernoulli)\\
    4. Expert Bias (Gaussian, then $s\tanh$)\\
    5. Verification (Bernoulli)\\[2pt]
    Inference thresholds gate probabilities\\
    at 0.5 and uses the Gaussian mean.\\
    A gate whose component is missing stays closed;\\
    ignored actions leave the policy log-probability.
};

\end{tikzpicture}
\end{document}
"""

def main():
    """Generate the TikZ LaTeX file."""
    output_file = "mantis_architecture.tex"

    with open(output_file, "w") as f:
        f.write(tikz_content)

    print(f"✓ Successfully created '{output_file}' in {os.getcwd()}")
    print("\nTo compile to PDF:")
    print("  pdflatex mantis_architecture.tex")
    print("\nTo render the README image (300 dpi):")
    print("  pdftoppm -png -r 300 -singlefile mantis_architecture.pdf mantis_architecture")
    print("\nOr upload to Overleaf for easy compilation.")

if __name__ == "__main__":
    main()
