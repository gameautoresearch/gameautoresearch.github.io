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

// What went into Gemini 4 Neon's $8.70 Polytopia recipe, the strongest 8 h retrain in budget.json (416 Elo),
// and its $2.90 recipe (330), read from their recipe.json / arch.json / network.py. The last column is the 8 h
// verdict from the single-technique comparisons on Opus 5.5, GPT-6.2 Sol and GPT-6.2 Astra recipes
// (reports/polytopia/techniques-combined-2026-11-19 in the bench repo), which predate Neon's runs.
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
const techniques: [string, string, boolean, boolean, string][] = [
  [
    "Playing old versions",
    "Practise against a pool of 20–24 past copies of itself, mostly the ones it still loses to",
    true,
    true,
    "Tested and dropped",
  ],
  [
    "Turn-based credit",
    "Measure time in game turns, not single commands",
    true,
    true,
    "up",
  ],
  [
    "Separate judge",
    "A smaller network that sees both sides of the board estimates who is winning",
    true,
    true,
    "Merging it hurt",
  ],
  [
    "Scoreboard inputs",
    "Hand the network a running tally of cities, units, territory and tech for both sides",
    true,
    false,
    "Not tested",
  ],
  [
    "Bigger network",
    "3–4 layers × 64 wide instead of the starter's 2 × 32",
    true,
    true,
    "even",
  ],
  [
    "Bigger learning steps",
    "3–4× the learning rate, lowered gradually over the run",
    true,
    true,
    "even",
  ],
  [
    "Reward for progress, faded out",
    "Pay for score and economy early, then shrink the bonus toward zero",
    true,
    false,
    "even",
  ],
];
const mark = (on: boolean) => (on ? "✓" : "—");
function TechniqueTable() {
  return (
    <Table
      head={[
        "Technique",
        "In plain words",
        "Neon $8.70 (416)",
        "Neon $2.90 (330)",
        "Earlier tests at 8 h",
      ]}
      rows={techniques.map(([name, plain, best, runnerUp, earlier]) => [
        <strong>{name}</strong>,
        plain,
        mark(best),
        mark(runnerUp),
        verdict(earlier),
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
        Top Line Results
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
            Standardized model scores on the two shared tasks (50 = average),
            with tabs for broader task pools.
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
        experiments on an NVIDIA L40S.
      </p>
      <p>Across our tasks, we used two submission formats:</p>

      <div className="ar-configs">
        <Cards
          items={[
            {
              title: "Submit a trained model",
              body: (
                <>
                  The agent submits model weights directly (as in Generals).
                  Research and training share one compute budget, so the agent
                  must weigh spending compute on tests and research against
                  saving it for the final training run.
                </>
              ),
            },
            {
              title: "Submit a training recipe",
              body: (
                <>
                  The agent submits training code and settings, and we retrain
                  the recipe from scratch (as in Pok&eacute;mon 3v3 and
                  Polytopia). Because the compute of the final run is fixed, the
                  agent&rsquo;s session budget goes purely to research.
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
            Best network trained over time on Polytopia from an empty file, with
            diamonds showing each final 15-minute retrain.
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
        How efficient are the recipes
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
        less compute it needs to train a network of equivalent strength to a
        basic PPO recipe. We were curious about the "payoff" timelines of tokens
        and compute spent on improving this recipe. Most runs do not pay back
        within 8 hours of training. The best pays back dramatically, though. One
        of {costMult.top.model}&rsquo;s sessions trains in 8 hours a network that our
        default recipe would need about {Math.round(costMult.top.starterHours)}{" "}
        hours to match ($
        {costMult.top.starterUsd}
        ), so it needs{" "}
        <strong>about {costMult.top.x.toFixed(0)}× less total compute</strong>.
        The next best, {costMult.second.model}
        &rsquo;s ${costMult.second.spent.toFixed(2)} session, reaches{" "}
        {costMult.second.x.toFixed(1)}× savings.
      </p>
      <CostMultiplier />

      <section className="ar-outlier" aria-labelledby="neon-outlier">
        <span className="ar-outlier-kicker">Outlier run</span>
        <h3 id="neon-outlier">
          How Gemini 4 Neon&rsquo;s $8.58 recipe pulled away
        </h3>
        <p>
          Using roughly half as many samples as the starter, this recipe matched
          the starter&rsquo;s 8-hour bot in 2 hours and finished at 416 Elo—72
          points clear of any other submission and stronger than the
          starter&rsquo;s 64-hour run. Major interventions that set it apart
          from other models&rsquo; submissions:
        </p>
        <ul className="ar-outlier-list">
          <li>
            <strong>Cheap full-board context:</strong> Added a 6×6 coarse
            spatial pyramid inside a 4×64 residual net, giving every tile
            board-wide reach without self-attention&rsquo;s speed penalty.
          </li>
          <li>
            <strong>Explicit scoreboard summaries:</strong> Computed 36
            visible-board tallies and relative leads (cities, units, HP,
            territory, tech, and estimated score).
          </li>
          <li>
            <strong>Omniscient critic distilled into the policy:</strong> Paired
            a separate training-only judge that sees both sides through the fog of
            war with an auxiliary value head that trains the policy&rsquo;s own
            trunk.
          </li>
          <li>
            <strong>Pruned historical self-play:</strong> Kept a 20-checkpoint
            opponent pool (25% of games) weighted toward past copies it still
            lost to, evicting easy ones—whereas other models dropped historical
            pools after short tests.
          </li>
        </ul>
      </section>

      <h2 className="ar-config ar-c3" id="taste">
        What separates strong researchers from weak
        ones?
      </h2>

      <h3 id="inside">Experimental discipline</h3>
      <p>
        Most models propose plausible changes and do experiments, but some
        models are dramatically more effective at choosing good interventions and
        interpreting results accurately.
      </p>
      <Figure
        n={3}
        wide
        caption={
          <>
            Pokémon 3v3 training runs over real session time, colored from
            weaker (magenta) to stronger (cyan) than the starter.
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
        We retrained every Polytopia recipe for 8 hours. Gemini 4 Neon's $8.70
        recipe was the strongest by a clear margin: 416 Elo, about 70 above the
        best recipe from any other model. It stacked several changes at once, so
        the table lists what went into it and into Neon's $2.90 recipe (330,
        tied for second). The last column shows what earlier one-at-a-time
        tests on Opus, Sol and Astra recipes found. Both Neon recipes practised
        against past copies of themselves, which the earlier models had tried
        and dropped, and both used turn-based credit, the only change in either
        recipe that also held up in those tests.
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
        n={4}
        caption={
          <>
            Forecast skill across {pokeSkill.items} Pokémon runs (0 = blind
            guess, 1 = perfect) compared with a ridge regression baseline
            (dotted line).
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
