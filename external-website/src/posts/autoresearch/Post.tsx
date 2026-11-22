import { Cards, Figure, Table } from "../shared/text";
import Showcase from "./Showcase";
import ResearchLoop from "./ResearchLoop";
import TopScore from "./TopScore";
import { ForecastLead, pokeSkill, polyForecastN } from "./Budget";
import { CostMultiplier, costMult } from "./CostMultiplier";
import { ResearchRace } from "./Frontier13";
import { SessionTimeline } from "./Timeline";
import "../shared/post.css";
import "./post.css";

// Setup and results sources: harness/README.md, the mid-size Generals contract,
// and the Pokémon 3v3 form, replication and retrain-noise readouts under reports/pokemon/.
// The Polytopia results are from the frontier6 and research-budget readouts under reports/polytopia/.
// The slow-simulator and forecast designs are documented in reports/generals/slowsim-2026-11-20.md
// and reports/forecast-skill-2026-11-19/PLAN.md.

// Polytopia dollar-budget recipes: each technique's effect after a 1 h and an 8 h retrain.
const VERDICT = {
  up: ["ar-v-up", "▲ Helps"],
  even: ["ar-v-even", "● About even"],
  down: ["ar-v-down", "▼ Hurts"],
} as const;
const verdict = (v: keyof typeof VERDICT | string) =>
  v in VERDICT ? (
    <span className={`ar-chip ${VERDICT[v as keyof typeof VERDICT][0]}`}>
      {VERDICT[v as keyof typeof VERDICT][1]}
    </span>
  ) : (
    v
  );
const techniques: [string, string, string, string][] = [
  ["Opportunity maps", "Show the bot what it can do on each tile", "up", "up"],
  [
    "Turn-based credit",
    "Measure time in game turns, not single commands",
    "up",
    "up",
  ],
  [
    "Smaller judge",
    "Shrink the network that estimates who is winning",
    "even",
    "even",
  ],
  ["Bigger learning steps", "Raise the learning rate about 3×", "up", "even"],
  [
    "Strategy hints",
    "Write Polytopia know-how into the bot by hand",
    "even",
    "down",
  ],
  ["Bigger network", "More layers or wider layers", "even", "even"],
  [
    "Reward for progress",
    "Pay the bot for score or economy along the way",
    "down",
    "even",
  ],
  ["One shared network", "Merge the player and the judge", "even", "down"],
  [
    "Playing old versions",
    "Practise against past copies of itself",
    "Lost every test",
    "Never submitted",
  ],
];
function TechniqueTable() {
  return (
    <Table
      head={["Technique", "In plain words", "After 1 hour", "After 8 hours"]}
      rows={techniques.map(([name, plain, early, late]) => [
        <strong>{name}</strong>,
        plain,
        verdict(early),
        verdict(late),
      ])}
    />
  );
}

export default function AutoresearchPost() {
  return (
    <div className="pt ar">
      <ResearchLoop />
      <p className="pt-lede">
        Can AI agents train strong domain-specific models? Agents have shown
        superhuman skills writing deterministic code. But related fields like
        machine learning pose different challenges: dealing with limited compute
        and noisy training signals. We gave AI models a GPU budget and tasked
        them with training efficient neural networks to play games like Pokémon,
        Polytopia, and Generals.io. The agents allocated their resources,
        implemented and tested ideas. We saw a wide range in taste, efficiency,
        and outcomes as they competed to produce the strongest final policies.
      </p>
      <Showcase />
      <p>
        As vibe-coding has led to the proliferation of custom software, so
        "vibe-training" promises new frontiers of specialized neural networks
        for niche domains. The efficiency of these solutions depends on the
        agent's judgment in executing machine learning workflows: carefully
        measuring success, choosing experiments, and allocating limited runs. We
        observed intelligent agents were able to train strong policies for
        smaller budgets.
      </p>

      <h2 className="ar-config ar-c1" id="results">
        <small>Results</small>Overall model scores
      </h2>
      <p>
        Nineteen models ran the same two tasks, Polytopia from an empty file and
        Pokémon 3v3 from a working starter. A group of six frontier models leads
        the rest, with little between them, and the same models lead when all
        eight research tasks across Polytopia, Pokémon, Hex, Generals and Clash
        are pooled.
      </p>
      <Figure
        n={1}
        wide
        caption={
          <>
            Two-task index: every model ran Polytopia from an empty file and
            Pok&eacute;mon 3v3 from a working starter (50 = average model).
            Whiskers span the two tasks taken alone. Gemini 4 Neon, GPT-6.2 Sol,
            Gemini 4 Carbon, GPT-6.1 Sol, GPT-6 Astra and Claude Fable 5.2 are
            within four points of each other. The other tabs pool all eight
            research tasks, where some models have only one or two.
          </>
        }
      >
        <TopScore />
      </Figure>

      <p>
        The headline score uses only the two tasks every model ran, so each
        model is judged on the same evidence. We standardize each task across
        the models and average the two: 50 is the average model, and 10 points
        is one standard deviation. The pooled tabs fit one rating through all
        eight research tasks, standardized within each task. The whiskers there
        show how much a score moves when any one task is left out, and models
        with fewer tasks have less evidence behind them. The forecasting tab
        adds a separate test of how well models predict training outcomes.
      </p>

      <h2 className="ar-config" id="setup">
        How we designed the benchmark
      </h2>
      <p>
        We wanted agents to make the decisions a researcher makes between runs.
        Which idea is worth testing? How long should it train? When is a
        practice result strong enough to bet the rest of the budget on it? Each
        agent got the same kind of workspace: a game rulebook, simulator code,
        editable training code and configs, and a command-line tool to launch
        experiments on an NVIDIA L40S, run practice matches, and submit its
        work.
      </p>
      <p>
        We kept practice and grading separate. Practice gave agents feedback
        they could act on during the session. The host evaluated the final
        policies on held-out games and played them against one another in a
        round robin. We also trained copies of the untouched starter so we could
        tell whether the field had improved on what it was given. The three
        setups changed what the agent handed in and how we charged for research.
      </p>

      <div className="ar-configs">
        <Cards
          items={[
            {
              id: "config-network",
              kicker: "Configuration 1",
              title: "Submit a trained network",
              body: (
                <>
                  Agents use one compute budget to experiment, train, and choose
                  a checkpoint. We grade the submitted weights directly, as a
                  team would evaluate a model before shipping it.
                  <br />
                  In Generals, agents had 4.5 GPU-hours and a compute limit on
                  each move.
                </>
              ),
            },
            {
              id: "config-recipe-time",
              kicker: "Configuration 2",
              title: "Submit a recipe under a GPU-time cap",
              body: (
                <>
                  Agents get a fixed number of GPU-hours for experiments and
                  submit their training code and settings. We retrain each
                  recipe from scratch to see how well it works when handed off.
                  <br />
                  In Pokémon 3v3, agents had four GPU-hours to improve a working
                  starter, followed by a one-hour retrain.
                </>
              ),
            },
            {
              id: "config-recipe-dollars",
              kicker: "Configuration 3",
              title: "Submit a recipe under a dollar budget",
              body: (
                <>
                  Agents spend a fixed cloud budget on experiments and submit a
                  training recipe. We retrain it and check whether the training
                  savings cover the cost of finding it.
                  <br />
                  In Polytopia, agents could spend up to $23 and rent two GPUs
                  at once.
                </>
              ),
            },
          ]}
        />
      </div>
      <Figure
        n={2}
        wide
        caption={
          <>
            Polytopia from an empty starter. Lines show the strongest network
            each agent had trained by that point. Diamonds show the fresh
            15-minute retrain of the final submitted recipe. Elo is relative to
            the 15-minute PPO starter.
          </>
        }
      >
        <ResearchRace />
      </Figure>
      <p>
        GPT-6 Astra&rsquo;s best in-session network edged out our tuned starter
        trained for 200 million samples. Its fresh retrain ranked first among
        submissions and stayed below the tuned starter.
      </p>

      <h2 className="ar-config ar-c2" id="efficiency">
        <small>Efficiency &amp; payback</small>How efficient are the recipes
        agents discover?
      </h2>
      <p>
        GPU experiments helped on average compared with the one-cent condition.
        Among agents with enough budget to run experiments, the spread between
        recipes at the same budget was much bigger than the difference between
        budget levels. Most $8.70 sessions spent their budget in about two
        hours, so giving them a longer deadline mostly gave us another attempt
        at the same spend.
      </p>
      <p>
        We measure a recipe by its <strong>compute multiplier</strong>: how much
        less compute it needs than our unmodified starter to train a network of
        the same strength. A better recipe only saves compute once it pays back
        the GPU hours spent finding it, so we retrained all {costMult.n} working
        recipes from our Polytopia budget sweep for 8 hours and charged each one
        its research spend on top of training. Most do not pay back within 8
        hours: only {costMult.above} of {costMult.n} are above 1×. The best pays
        back dramatically, though. {costMult.top.model}&rsquo;s $
        {costMult.top.spent.toFixed(2)} session trains in 8 hours a network that
        our starter would need about {Math.round(costMult.top.starterHours)}{" "}
        hours to match ($
        {costMult.top.starterUsd}
        {costMult.top.extrap
          ? ", extrapolated past our starter's 64-hour run"
          : ""}
        ), so it needs{" "}
        <strong>about {costMult.top.x.toFixed(0)}× less compute</strong> even
        with its research paid for. The next best, {costMult.second.model}
        &rsquo;s ${costMult.second.spent.toFixed(2)} session, reaches{" "}
        {costMult.second.x.toFixed(1)}×.
      </p>
      <Figure
        n={3}
        wide
        caption={
          <>
            Compute multiplier against total cost on Polytopia: the
            starter&rsquo;s training cost to reach the same Elo, divided by the
            recipe&rsquo;s research GPU spend plus host training on one L40S at
            $2.90/h (model tokens excluded). Above 1×, a recipe has paid for its
            own research. Open points are stronger than anything in the
            starter&rsquo;s 64-hour run, so their multiplier extends the
            starter&rsquo;s fitted curve. &ldquo;Training compute only&rdquo;
            drops the research spend: each bar is a model&rsquo;s best recipe,
            measured on training time alone and averaged over its 6, 7 and 8
            hour checkpoints so that one lucky checkpoint cannot set the bar;
            the whisker on an extrapolated bar is the 95% range at 8 hours.
          </>
        }
      >
        <CostMultiplier />
      </Figure>

      <h2 className="ar-config ar-c3" id="taste">
        <small>Research taste</small>What separates strong researchers from weak
        ones?
      </h2>

      <h3 id="inside">Experimental discipline</h3>
      <p>
        Most models propose plausible changes; what separates strong researchers
        from weak ones is how they test them. In Pokémon, GPT-6.1 Sol and GPT-6
        Astra launch full-length 1-hour runs matching the graded final, let them
        finish, evaluate once, and move on—using 2–4M prompt tokens per session.
        Weaker models average 30–80M tokens thrashing through aborted 5–20
        minute probes that do not predict 1-hour performance.
      </p>
      <Figure
        n={4}
        wide
        caption={
          <>
            Every Pokémon training run at its real session time, colored by
            private strength (cyan = beats starter, magenta = loses to starter).
            Crosses are aborted runs; diamonds are practice evals; flags are
            submissions.
          </>
        }
      >
        <SessionTimeline />
      </Figure>

      <h3 id="tools">Would agents fix a slow simulator?</h3>
      <p>
        Fast simulation is usually part of the workspace we give agents. In a
        separate Generals test we supplied a plain Python simulator that played
        one game at a time and gave no hints about its speed. We checked that it
        matched the fast host engine on 120,000 game ticks. Practice, grading
        and the tournament still ran on the host engine, so agents could rebuild
        their training simulator without changing the game their policies were
        judged on.
      </p>
      <p>
        All four agents noticed the bottleneck. Claude Opus 5.5 and GPT-6.1 Sol
        built batched simulators, checked their behavior against the supplied
        implementation, and trained at roughly 14,600 and 5,400 game ticks per
        second. DeepSeek V4 Pro and Kimi K3.1 Pro stayed with the supplied
        engine at about 565 and 630 ticks per second. The two rebuilders took
        the top two tournament spots. Their recipes differed in other ways too,
        so the tournament shows how the whole strategy worked within the budget.
      </p>

      <h3 id="techniques">What changes actually held up</h3>
      <p>
        Retraining Polytopia recipes at both 1 hour and 8 hours separated
        short-run tricks from durable gains: better game features held up at 8
        hours, while higher learning rates and hand-coded strategy hints faded
        or hurt late.
      </p>
      <TechniqueTable />

      <h3 id="forecast">Predicting which run will win</h3>
      <p>
        Forecasting lets us examine a decision researchers make before they buy
        compute. We included both fresh finals and individual research runs, and
        showed forecasters only evidence available at launch. The simpler
        benchmark predicts from run length and prior practice scores. We fit it
        on other sessions and resampled whole sessions when estimating
        uncertainty, because runs from the same researcher session share
        history.
      </p>
      <p>
        Good research also requires predicting which experiments will pay off—a
        place where current models are still weak. Given {pokeSkill.items}{" "}
        anonymized Pokémon runs cut off at launch, all {pokeSkill.aboveBlind} of{" "}
        {pokeSkill.n} models beat a blind guess, but only{" "}
        {pokeSkill.above.length} ({pokeSkill.above.join(" and ")}) reliably beat
        a simple ridge regression on run length and practice history. Given
        1-hour evidence for four Polytopia recipes, all {polyForecastN} models
        ranked the eventual 8-hour winner last.
      </p>
      <Figure
        n={5}
        caption={
          <>
            Forecast skill across {pokeSkill.items} Pokémon runs (0 = blind
            guess, 1 = perfect; 90% intervals over sessions). Dotted line: ridge
            regression
          </>
        }
      >
        <ForecastLead />
      </Figure>

      <h2 className="ar-config" id="closing-thoughts">
        Closing Thoughts
      </h2>
      <p>
        In a world where software truly proliferates, writing code is the first
        step. We expect training models to become just as routine. An agent
        could build an application, train the models it needs, and ship them to
        run independently. Autonomous training is a building block for recursive
        self-improvement. It could also put a specialized model within reach for
        almost any niche task, from a crop-image classifier to a small model
        embedded in clinical software. We built this benchmark to measure how
        well agents can do that training under a compute budget.
      </p>
      <p>
        Take games, for example. Rules can change daily during production and
        keep changing after a live-service game launches. If agents can cheaply
        retrain and test bots as the rules change, studios can use them to check
        game balance and give players reliable teammates. That could make both
        practical for more developers, especially indie teams with little time
        to spend maintaining bots.
      </p>
      <p>We thank Modal for providing compute credits for this research.</p>

      <hr />
      <h3 id="related">Related work</h3>
      <ul className="ar-credits">
        <li>
          Andy L. Jones,{" "}
          <a
            href="https://arxiv.org/abs/2104.03113"
            target="_blank"
            rel="noreferrer"
          >
            Scaling Scaling Laws with Board Games
          </a>
          , arXiv:2104.03113, 2021.
        </li>
        <li>
          Ben Rank, Hardik Bhatnagar, Ameya Prabhu, Shira Eisenberg, Karina
          Nguyen, Matthias Bethge and Maksym Andriushchenko,{" "}
          <a
            href="https://arxiv.org/abs/2603.08640"
            target="_blank"
            rel="noreferrer"
          >
            PostTrainBench: Can LLM Agents Automate LLM Post-Training?
          </a>
          , arXiv:2603.08640, 2026.
        </li>
        <li>
          Elie Bakouch and Prime Intellect,{" "}
          <a
            href="https://www.primeintellect.ai/blog/measuring-autonomous-research"
            target="_blank"
            rel="noreferrer"
          >
            Measuring Autonomous AI Research
          </a>
          , Prime Intellect blog, August 2026.
        </li>
        <li>
          Kevin Zhang,{" "}
          <a
            href="https://thekevinz.com/blog/polytopia-bot"
            target="_blank"
            rel="noreferrer"
          >
            Building a Superhuman Polytopia bot using self-play RL
          </a>
          , blog post, October 2026.
        </li>
      </ul>

      <h3 id="credits">Open-source credits</h3>
      <p>
        Several of our simulators are built on, or checked against, open-source
        projects. Thank you to their authors. We keep each project&rsquo;s
        license and notices with our copy.
      </p>
      <ul className="ar-credits">
        <li>
          <strong>Pokémon:</strong>{" "}
          <a
            href="https://github.com/smogon/pokemon-showdown"
            target="_blank"
            rel="noreferrer"
          >
            Pokémon Showdown
          </a>{" "}
          by Guangcong Luo and contributors (MIT), the battle simulator;{" "}
          <a
            href="https://github.com/hsahovic/poke-env"
            target="_blank"
            rel="noreferrer"
          >
            poke-env
          </a>{" "}
          by hsahovic (MIT); and the battle code of the{" "}
          <a
            href="https://github.com/smogon/pokemon-showdown-client"
            target="_blank"
            rel="noreferrer"
          >
            Pokémon Showdown client
          </a>
          , which plays the replays on this page.
        </li>
        <li>
          <strong>Polytopia:</strong> the Rust engine of{" "}
          <a
            href="https://github.com/HenBOMB/Polyfish"
            target="_blank"
            rel="noreferrer"
          >
            Polyfish
          </a>{" "}
          by HenBOMB (Apache-2.0), which our GPU rule ports derive from and are
          validated against.
        </li>
        <li>
          <strong>Generals:</strong>{" "}
          <a
            href="https://github.com/strakam/generals-bots"
            target="_blank"
            rel="noreferrer"
          >
            generals-bots
          </a>{" "}
          by Matej Straka (MIT).
        </li>
        <li>
          <strong>Hex:</strong> MoHex 2.0 from{" "}
          <a
            href="https://github.com/cgao3/benzene-vanilla-cmake"
            target="_blank"
            rel="noreferrer"
          >
            Benzene
          </a>{" "}
          (LGPL-3.0), our evaluation opponent.
        </li>
        <li>
          <strong>Clash:</strong>{" "}
          <a
            href="https://github.com/voonhous/crforge"
            target="_blank"
            rel="noreferrer"
          >
            crforge
          </a>{" "}
          (Apache-2.0), the rules oracle for our C++/CUDA engine.
        </li>
        <li>
          <strong>Tavern (Battlegrounds):</strong>{" "}
          <a
            href="https://github.com/MCG-NJU/AutoTavern"
            target="_blank"
            rel="noreferrer"
          >
            AutoTavern
          </a>{" "}
          by the AutoTavern authors (MIT).
        </li>
        <li>
          <strong>Dou Dizhu:</strong>{" "}
          <a
            href="https://github.com/kwai/DouZero"
            target="_blank"
            rel="noreferrer"
          >
            DouZero
          </a>{" "}
          by Kuaishou (Apache-2.0), the rules oracle for our native engine.
        </li>
      </ul>
      <p
        className="pb-notes"
        style={{ fontSize: 13.5, color: "var(--secondary)" }}
      >
        Pokémon, The Battle of Polytopia, Generals.io, Clash Royale and
        Hearthstone Battlegrounds are trademarks of their respective owners.
        This project is not affiliated with or endorsed by them.
      </p>
    </div>
  );
}
