import { Fragment } from "react";
import { PAGE } from "./meta";
import AutoresearchPost from "./posts/autoresearch/Post";
import Toc from "./Toc";

function displayDate(date: string) {
  return new Date(`${date}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

// The whole site is this one post: no site nav, no index, no footer; a contents list sits in the left margin.
export default function App() {
  return (
    <div className="page-shell">
      <Toc rootId="post-body" />
      <div className="site-layout">
        <main className="profile">
          <article className="blog-post" aria-labelledby="post-title">
            <h1 id="post-title">{PAGE.title}</h1>
            <p className="post-byline">
              {PAGE.authors.map(({ name, url }, index) => (
                <Fragment key={url}>
                  {index > 0 && " and "}
                  <a href={url} rel="author">{name}</a>
                </Fragment>
              ))}
              {" · "}<time dateTime={PAGE.date}>{displayDate(PAGE.date)}</time>
            </p>
            <div className="post-content" id="post-body"><AutoresearchPost /></div>
          </article>
        </main>
      </div>
    </div>
  );
}
