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
        observed intelligent agents were able to train stronger policies for
        smaller budgets.
      </p>

      <h2 className="ar-config ar-c1" id="results">
        Overall Ranking
      </h2>
      <p>
        We graded 10 models across 5 model training tasks: Polytopia from an
        empty file, Pokémon 3v3 from a working starter, Polytopia recipes
        written under varying research budgets, forecasting training run results
        based on previous data, and a Generals network trained from an empty
        starter and ranked in a 19-network tournament.
      </p>
      <Figure
        n={1}
        wide
        caption={
          <>
            Overall model scores on five tasks with normalized scores, ±1
            standard error across tasks. Gemini 4 Carbon was not in the Generals
            tournament; its score averages its four other tasks.
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
        strong enough to bet the rest of the budget on it? Each agent got a
        workspace with a game rulebook, simulator code, editable training code
        and configs, and a command-line tool to launch experiments on an NVIDIA
        L40S and read logs.
      </p>
      <p>Each task uses one of two submission formats:</p>

      <SubmissionFormats />
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
        Over their 5 GPU-hours, GPT-6.2 Sol trained 28 networks and GPT-6 Astra
        trained 25, about one every 12 minutes. The best in-session networks of
        both edged out our tuned starter trained for 200 million samples.
        Astra&rsquo;s fresh retrain ranked first among submissions and stayed
        below the tuned starter.
      </p>

      <h2 className="ar-config ar-c2" id="efficiency">
        Can agents save money by spending more on research?
      </h2>
      <p>
        We were curious about the payoff of tokens and compute invested in
        improving the <strong>starter recipe</strong>, the basic RL setup every
        agent began from. We measure a recipe by its{" "}
        <strong>compute multiplier</strong>: how much <i>less</i> compute it
        needs to train a network as strong as one from the starter recipe. Most
        runs do not recoup their investment within 8 hours of training. Some pay
        back dramatically, though. One of {costMult.top.model}&rsquo;s sessions
        trains in 8 hours a network that the starter recipe would need about{" "}
        {Math.round(costMult.top.starterHours)} hours to match ($
        {costMult.top.starterUsd}
        ), so it needs{" "}
        <strong>about {costMult.top.x.toFixed(0)}× less total compute</strong>.
        The next, a {costMult.second.model} run that spent $
        {costMult.second.spent.toFixed(2)} on research, reaches{" "}
        {costMult.second.x.toFixed(1)}× savings after 8 hours of training.
      </p>
      <CostMultiplier />

      <section className="ar-outlier" aria-labelledby="neon-outlier">
        <span className="ar-outlier-kicker">Outlier run</span>
        <h3 id="neon-outlier">
          How Gemini 4 Neon&rsquo;s $8.58 recipe pulled away
        </h3>
        <p>
          After the same 8 hours of training, this recipe finished well ahead of
          every other submission. Four changes set it apart:
        </p>
        <NeonOutlierDiagrams />
      </section>

      <h2 className="ar-config ar-c3" id="taste">
        What do strong researchers do differently?
      </h2>
      <p>
        Most models propose plausible changes and do experiments, but some
        models are dramatically more effective at choosing good interventions
        and interpreting results accurately.
      </p>
      <p>
        Gemini 4 Neon&rsquo;s $8.58 Polytopia session&mdash;the same run that
        produced the 416 Elo outlier recipe above&mdash;is a good example. It
        ran <strong>five short A/B tests</strong> (mostly pairs of 12-minute
        runs for ~$0.60 each).
      </p>
      <Figure n={3} wide caption={<>Neon's 5 A/B tests</>}>
        <NeonDiscipline />
      </Figure>
      <p>
        Other models spent the same $8.70 Polytopia budget very differently.
        Where Gemini 4 Neon ran 11 experiments across 4.4 hours&mdash;short
        probes (5&ndash;13 min) followed by a 1-hour validation&mdash;GPT-6.2
        Astra, GPT-6.2 Sol and Kimi K3.1 Pro each put the whole budget into
        three ~1-hour runs and finished in about two hours, while Claude Fable
        5.2, Claude Opus 5.5 and DeepSeek V4.1 Flash sat in between with
        4&ndash;6 runs.
      </p>
      <Figure
        n={4}
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

      <h3 id="tools">Would agents fix a slow simulator?</h3>
      <p>
        Fast simulation is usually part of the workspace we give agents. In some
        tests, we supplied an accurate but single-threaded Python simulator.
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

      <h3 id="forecast">Predicting which run will win</h3>
      <p>
        Forecasting is a cheap microcosm of researchers&rsquo; ability: making
        accurate predictions before buying compute. We sampled several
        historical research runs, then showed forecasting agents only evidence
        from up until each run launched, like the training recipe and results
        from prior runs.
      </p>
      <p>
        Given {pokeSkill.items} anonymized Pokémon runs cut off at launch, only{" "}
        {pokeSkill.above.join(" and ")} reliably beat a simple ridge regression
        on run length and practice history.
      </p>
      <Figure
        n={5}
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
        We also ran older model releases from OpenAI, Anthropic, and Google on
        the same empty-starter Polytopia task to see how quickly training skill
        is advancing. Earlier in the year, frontier models struggled to produce
        useful policies at all&mdash;the oldest ones could not beat random play,
        and even midsummer releases fell well short of our basic PPO starter.
        Starting in the fall, newer models crossed a clear threshold and began
        training policies that beat the starter on the same compute budget.
      </p>
      <Figure
        n={6}
        wide
        caption={
          <>
            Compute multiplier of each model&rsquo;s fresh 15-minute Polytopia
            retrain by release date (February&ndash;November 2026). Higher means
            the unmodified PPO starter needs more compute to match the policy;
            points in the bottom band fall below the starter curve&rsquo;s
            measured floor. Each point is a single run; hover any point for
            details.
          </>
        }
      >
        <CapabilityTimeline />
      </Figure>

      <h2 className="ar-config" id="closing-thoughts">
        Closing Thoughts
      </h2>
      <p>
        In a world of cheap software, writing code is only the first step.
        Specialized models are coming into reach for almost any niche task, from
        image classifiers to motor controls. We built this benchmark to measure
        how well agents can do that training under a compute budget. We view
        this ability to train small useful models as an indicator of recursive
        self-improvement capabilities.
      </p>
      <p>
        At game studios like Curio, rules can change daily during production and
        keep changing after a live-service game launches. If we can cheaply
        retrain and test bots as the rules change, it helps check game balance
        and give players reliable teammates. This is newly practical for more
        developers, especially indie teams with little time to spend maintaining
        bots.
      </p>
      <p>
        We thank Modal for providing compute and services that made this
        research possible.
      </p>
      <a
        className="ar-sponsor"
        href="https://modal.com"
        target="_blank"
        rel="noreferrer"
      >
        <img src={modalWordmark} alt="Modal" width={140} />
      </a>

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
    </div>
  );
}
