import { Figure } from "../shared/text";
import Showcase from "./Showcase";
import ResearchLoop from "./ResearchLoop";
import SubmissionFormats from "./SubmissionFormats";
import TopScore from "./TopScore";
import { ForecastLead, pokeSkill } from "./Budget";
import { CostMultiplier, costMult } from "./CostMultiplier";
import NeonOutlierDiagrams from "./NeonOutlierDiagrams";
import NeonDiscipline from "./NeonDiscipline";
import RecipeTimelines from "./RecipeTimelines";
import { ResearchRace } from "./Frontier13";
import CapabilityTimeline from "./CapabilityTimeline";
import modalWordmark from "./modal-wordmark.svg";
import "../shared/post.css";
import "./post.css";

// Setup and results sources: harness/README.md, the mid-size Generals contract,
// and the Pokémon 3v3 form, replication and retrain-noise readouts under reports/pokemon/.
// The Polytopia results are from the frontier6 and research-budget readouts under reports/polytopia/.
// The slow-simulator and forecast designs are documented in reports/generals/slowsim-2026-11-20.md
// and reports/forecast-skill-2026-11-19/PLAN.md.

export default function AutoresearchPost() {
  return (
    <div className="pt ar">
      <ResearchLoop />
      <p className="pt-lede">
        Can AI agents train strong domain-specific models? Agents have shown
        superhuman skills writing deterministic code. But effective machine
        learning poses different challenges: limited compute and noisy training
        signals.
      </p>
      <p className="pt-lede">
        We gave AI models a GPU budget and tasked them with training
        efficient neural networks to play games like Pokémon, Polytopia, and
        Generals.io. The agents had to manage their budget to implement and
        test ideas. We saw a wide range in taste and efficiency, reflected in
        the strength of the policies they were able to train. Smarter, newer
        models trained much more powerful networks.
      </p>
      <Showcase />

      <h2 className="ar-config ar-c1" id="results">
        Overall Ranking
      </h2>
      <p>
        We evaluated ten models across five training tasks: Two variants of
        Polytopia, Pokémon Showdown, Generals, as well as a task based on
        forecasting the success of a training recipe without running it.
      </p>
      <Figure
        n={1}
        wide
        caption={
          <>
            Overall model scores on five tasks with normalized scores, ±1
            standard error across tasks.
          </>
        }
      >
        <TopScore />
      </Figure>

      <h2 className="ar-config" id="setup">
        Benchmark Design
      </h2>
      <p>
        We wanted agents to manage the process like a researcher. Which idea is
        worth testing? How long should it train? When is a practice result
        strong enough to bet the rest of the budget on it? Their workspace
        provided:
      </p>
      <ul>
        <li>
          <strong>Compute via Modal.</strong> A fixed budget and a command-line
          tool to launch jobs on NVIDIA L40S workers and read their logs.
        </li>
        <li>
          <strong>Complete game rulebook.</strong> The rules, legal moves, and
          scoring for the task.
        </li>
        <li>
          <strong>A simulator.</strong> Most tasks came with an optimized
          simulator. In one Generals variant, agents got a slow, single-threaded
          simulator they could improve themselves.
        </li>
        <li>
          <strong>Editable training code and configs.</strong> Depending on the
          task, agents started with a working training recipe or a scaffold
          without learning code.
        </li>
        <li>
          <strong>A practice suite.</strong> Agents could evaluate their
          checkpoints during the session and use the results to decide what to
          try next or what to submit.
        </li>
      </ul>
      <p>
        Agents design and submit a <strong>training recipe</strong>. They use
        their research budget on iterating on this code, then we retrain the
        recipe from scratch on a separate, fixed training budget and grade the
        network it produces.
      </p>

      <SubmissionFormats />
      <p>
        We landed on this format because we saw that balancing small compute
        budgets between research and training incentivized very simple
        strategies. Standardizing the training budget incentivized more
        exploration, and is a better match for how scaled-up training works:
        many small experiments which inform the recipe for an expensive "yolo"
        run.
      </p>
      <Figure
        n={2}
        wide
        caption={
          <>
            5-hour autoresearch session on a 15-minute Polytopia training recipe
            from an empty file. Measured via win rate against the 15-minute
            trained baseline recipe.
          </>
        }
      >
        <ResearchRace />
      </Figure>
      <p>
        Over a single five-hour session, the strongest agents iterated
        relentlessly&mdash;GPT-6.2 Sol and GPT-6 Astra each trained over two
        dozen networks. Their best 15-minute-trained networks beat a reference
        model that was trained for more than twice as long (36 minutes).
      </p>

      <h2 className="ar-config ar-c2" id="efficiency">
        Can agents save money by spending more on research?
      </h2>
      <p>
        We were curious whether compute spent to improve the{" "}
        <strong>starter recipe</strong> actually pays for itself. We measure
        each recipe by its <strong>compute multiplier</strong>: how much{" "}
        <i>less</i> compute it needs to match the starter&rsquo;s strength, so
        crossing the &ldquo;Starter 1×&rdquo; line means the research spend has
        recouped its cost. Most sessions stay below that line over an 8-hour
        training run, while a few break even and one pulls far ahead: a{" "}
        {costMult.top.model} session that spent ${costMult.top.spent.toFixed(2)}{" "}
        on research reaches the same strength with{" "}
        <strong>about {costMult.top.x.toFixed(0)}× less total compute</strong>.
      </p>
      <Figure
        n={3}
        wide
        caption={
          <>
            Compute multiplier of researched Polytopia recipes relative to the
            unmodified starter (1×), with research spend included on the left
            and training compute only on the right.
          </>
        }
      >
        <CostMultiplier />
      </Figure>

      <section className="ar-outlier" aria-labelledby="neon-outlier">
        <span className="ar-outlier-kicker">Outlier run</span>
        <h3 id="neon-outlier">What set Gemini 4 Neon&rsquo;s recipe apart</h3>
        <p>
          Neon leaned into Polytopia&rsquo;s structure with a compact spatial
          map, partial observability under fog of war, countable economic state,
          and evolving tactical counters. These are well-known techniques. Neon
          seemed to have an intuition for which ones would work well together
          in this game. Across repeated runs of the same task, no other model
          found a recipe with a comparable gain in compute efficiency.
        </p>
        <NeonOutlierDiagrams />
      </section>

      <h2 className="ar-config ar-c3" id="taste">
        What do strong researchers do differently?
      </h2>
      <p>
        Most models propose plausible ideas and run experiments, yet some are
        far better at choosing high-impact interventions and reading noisy
        signals accurately. Rather than gambling its budget on a few big
        rewrites, Neon for instance assembled the winning recipe above through{" "}
        <strong>five paired A/B tests</strong>
        &mdash;short, inexpensive probes to confirm each change before spending
        a final validation run on the combined stack.
      </p>
      <Figure n={4} wide caption={<>Neon&rsquo;s 5 A/B tests</>}>
        <NeonDiscipline />
      </Figure>
      <p>
        Comparing the seven models side by side under the same Polytopia
        research budget makes this gap in experimental style clear. Most agents
        ran a handful of long runs and blew their budget. Neon allocated many
        fast, cheap probes&mdash;testing far more hypotheses before committing
        to a full validation run.
      </p>
      <Figure
        n={5}
        wide
        caption={
          <>
            How all seven models in the $8.70 Polytopia recipe task spent their
            budget. Right columns show completed GPU training runs and 8-hour
            retrain Elo.
          </>
        }
      >
        <RecipeTimelines />
      </Figure>
      <p>
        <strong>
          Good AI research is often about finding ways to sample faster.
        </strong>{" "}
        Whether by shrinking experiments into cheap, rapid probes, cutting
        overhead from the training loop, or accelerating the simulator itself,
        the strongest agents buy themselves more feedback per dollar.
      </p>

      <h3 id="tools">Would agents improve the simulator itself?</h3>
      <p>
        In our own experience{" "}
        <a
          href="https://thekevinz.com/blog/polytopia-bot"
          target="_blank"
          rel="noreferrer"
        >
          training Polytopia bots
        </a>
        , simulator speed was one of the biggest bottlenecks to iteration and
        cost. We provided fast simulators in most of the workspaces we give
        agents, but we also tested a variant where we provided only an accurate
        but slow, single-threaded Python simulator to see whether agents would
        optimize it themselves.
      </p>
      <p>
        All four agents in this experiment noticed the bottleneck. Claude Opus
        5.5 and GPT-6.1 Sol rebuilt the simulator to run batched, verified it
        against the original, and trained on roughly 10× to 25× more samples for
        the same compute. DeepSeek V4 Pro and Kimi K3.1 Pro stayed with the
        supplied engine. The two rebuilders took the top two spots in the
        tournament.
      </p>

      <h3 id="forecast">
        Can models predict training outcomes without actually running them?
      </h3>
      <p>
        Evaluating agents on full RL sessions is expensive, so we wanted a
        cheaper signal. Forecasting tests a core piece of research
        judgment&mdash;predicting which experiment will pay off before buying
        compute&mdash;without needing any GPUs at all. We took historical
        research runs and showed each model only the evidence available before a
        run launched, like its training recipe and prior results. Even at a
        fraction of the cost, forecasting skill tracked which models performed
        best in full training runs.
      </p>
      <p>
        Only {pokeSkill.above.join(" and ")} reliably beat a simple regression
        on run length and practice history.
      </p>
      <Figure
        n={6}
        caption={
          <>
            Forecast skill across {pokeSkill.items} Pokémon runs (0 = blind
            guess, 1 = perfect) compared with a simple baseline.
          </>
        }
      >
        <ForecastLead />
      </Figure>

      <h2 className="ar-config" id="progress">
        How are model training capabilities improving over time?
      </h2>
      <p>
        We also ran older model releases on the same empty-starter Polytopia
        task to see how quickly training skill is advancing. Frontier models
        from early 2026 struggled to produce useful policies at all&mdash;
        February releases could not beat random play, and August releases fell
        well short of our basic PPO starter recipe. Starting in the fall, newer
        models crossed a clear threshold and began training policies that beat
        the starter on the same or much lower compute budget.
      </p>
      <Figure
        n={7}
        wide
        caption={
          <>
            Compute multiplier of each model&rsquo;s best in-session network, by
            release date (February&ndash;November 2026). Higher means more
            compute-efficient for the same strength.
          </>
        }
      >
        <CapabilityTimeline />
      </Figure>

      <h2 className="ar-config" id="closing-thoughts">
        Closing Thoughts
      </h2>
      <p>
        Vibe-coding made it easier to build software for your own needs. We
        encourage you to vibe-train: If you have a niche task and a way to
        measure success, give a coding agent a modest compute budget and ask
        it to train and evaluate a model for you.
      </p>
      <p>
        For turn-based, fixed-action-space games with fast simulators like
        Polytopia and Generals, our results show you can already
        &ldquo;vibe-train&rdquo; a strong, human-competitive RL policy for roughly{" "}
        <strong>$50&ndash;$100</strong> in combined research and training
        compute. At game studios like Curio, where rules can shift daily in
        production and keep evolving after a live-service launch, cheaply
        retraining bots as the game changes helps check balance and give players
        reliable teammates&mdash;something newly practical even for small indie
        teams.
      </p>
      <p>
        That price point also marks the boundary of where today&rsquo;s
        vibe-training works out of the box. Real-time games like StarCraft:
        Brood War, Dota, or CS:GO sit on the far side of that spectrum:
        continuous or sprawling action spaces, real-time control, and heavier,
        slower simulators mean each sample costs far more to generate and
        policies need orders of magnitude more of them, pushing total costs{" "}
        <strong>100&ndash;1,000× higher</strong>. Mapping out that
        cost-versus-game-complexity curve&mdash;and measuring how far
        agent-driven algorithmic and simulator improvements can bend it
        downward&mdash;is a natural direction for future work.
      </p>
      <p>
        More broadly, we built this benchmark to measure how well agents
        allocate scarce compute and learn from noisy experiments. As models
        advance from failing to beat random play to discovering multi-fold
        compute multipliers in a single session, their ability to train small,
        useful models shows support for recursive self-improvement capability.
      </p>
      <aside className="ar-sponsor">
        <a href="https://modal.com" target="_blank" rel="noreferrer">
          <img src={modalWordmark} alt="Modal" width={120} />
        </a>
        <p>
          We thank Modal for supporting this research. The evaluation used about
          3,000 GPU-hours and an estimated 30,000 CPU core&#8209;hours.
        </p>
      </aside>

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
        Several of our simulators are built on or checked against the following
        open-source projects. Thank you to the authors of:
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
      </ul>
      <p
        className="pb-notes"
        style={{ fontSize: 13.5, color: "var(--secondary)" }}
      >
        Pokémon, The Battle of Polytopia and Generals.io are trademarks of their
        respective owners. This project is not affiliated with or endorsed by
        them.
      </p>

      <h3 id="trajectories">Agent trajectories</h3>
      <p>
        <a href={`${import.meta.env.BASE_URL}trajectories`}>
          Browse the full trajectory viewer →
        </a>{" "}
      </p>
    </div>
  );
}
