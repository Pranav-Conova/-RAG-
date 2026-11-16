/* AsOf web UI: ask a question as of a date, optionally compare AsOf with a baseline RAG.
 *
 * Vanilla JS, no dependencies. All model and source text is inserted with textContent /
 * text nodes (never parsed as HTML), so nothing returned by the API can inject markup.
 */
(function () {
  "use strict";

  var form = document.getElementById("ask-form");
  if (!form) return;

  var qEl = document.getElementById("q");
  var dateEl = document.getElementById("as-of");
  var cmpEl = document.getElementById("compare");
  var btn = document.getElementById("ask-btn");
  var statusEl = document.getElementById("status");
  var resultsEl = document.getElementById("results");
  var endpoint = form.getAttribute("data-endpoint") || "/api/ask";

  var TIMEOUT_MS = 120000;
  var SLOW_HINT_MS = 8000;
  var MIN_DATE = "2017-07-01";
  var MODES = {
    asof: { label: "AsOf", sub: "Time-aware: only text in force on the date" },
    baseline: { label: "Baseline RAG", sub: "Same corpus and model, no date filtering" }
  };
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var KIND_LABELS = {
    act: "Act", rules: "Rules", notification: "Notification", circular: "Circular",
    order: "Order", other: "Document"
  };
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var run = null; // the current request set: {id, controllers, results, refDate, userDate}
  var runCounter = 0;

  // ------------------------------------------------------------------ DOM helpers

  /** Create an element. attrs: class, text, on<event>, or plain attributes. Children: nodes or strings. */
  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) return;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = String(v);
        else if (k.slice(0, 2) === "on" && typeof v === "function") node.addEventListener(k.slice(2), v);
        else node.setAttribute(k, v === true ? "" : String(v));
      });
    }
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }

  function append(node, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { append(node, c); }); return; }
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("error", !!isError);
  }

  // ------------------------------------------------------------------ dates

  function isISODate(s) { return typeof s === "string" && /^\d{4}-\d{2}-\d{2}$/.test(s); }

  function todayISO() {
    var d = new Date();
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" +
      String(d.getDate()).padStart(2, "0");
  }

  /** "2020-02-15" -> "15 Feb 2020" without timezone surprises. */
  function fmtDate(s) {
    if (!isISODate(s)) return s ? String(s) : "";
    var p = s.split("-");
    return parseInt(p[2], 10) + " " + MONTHS[parseInt(p[1], 10) - 1] + " " + p[0];
  }

  /** Where a hit stands relative to the reference date. Validity is [valid_from, valid_to). */
  function validity(hit, refDate) {
    var from = isISODate(hit.valid_from) ? hit.valid_from : null;
    var to = isISODate(hit.valid_to) ? hit.valid_to : null;
    var lower = from || (isISODate(hit.issue_date) ? hit.issue_date : null);
    var range = (from || to) ? "valid " + (from ? fmtDate(from) : "?") + " → " +
      (to ? fmtDate(to) : "present") : "";
    if (!refDate) return { state: "unknown", cls: "", label: range || "Validity not recorded", range: "", stale: false };
    if (lower && lower > refDate) {
      return { state: "future", cls: "bad", label: "Not yet in force on " + fmtDate(refDate),
               range: range || "issued " + fmtDate(lower), stale: true };
    }
    if (to && to <= refDate) {
      return { state: "superseded", cls: "bad", label: "Superseded on " + fmtDate(to), range: range, stale: true };
    }
    if (from || to) {
      return { state: "inforce", cls: "ok", label: "In force on " + fmtDate(refDate), range: range, stale: false };
    }
    return { state: "unknown", cls: "", label: "Validity not recorded", range: "", stale: false };
  }

  // ------------------------------------------------------------------ text helpers

  /** Lowercased, whitespace-collapsed copy of s plus a map from its indices to s's indices. */
  function normWithMap(s) {
    var out = "", map = [], prevSpace = true;
    for (var i = 0; i < s.length; i++) {
      var ch = s[i];
      if (/\s/.test(ch)) {
        if (!prevSpace) { out += " "; map.push(i); prevSpace = true; }
      } else {
        out += ch.toLowerCase(); map.push(i); prevSpace = false;
      }
    }
    return { text: out, map: map };
  }

  function cleanQuote(q) {
    return String(q || "").replace(/^[\s"'“”‘’.…]+|[\s"'“”‘’.…]+$/g, "")
      .replace(/\s+/g, " ").toLowerCase();
  }

  /** Character ranges [start, end) of each quote found (case/space-insensitively) in text. */
  function findQuotes(text, quotes) {
    var n = normWithMap(text), ranges = [];
    quotes.forEach(function (q) {
      var cq = cleanQuote(q);
      if (cq.length < 4) return;
      var at = n.text.indexOf(cq), len = cq.length;
      if (at < 0 && cq.length > 80) { cq = cq.slice(0, 80); at = n.text.indexOf(cq); len = cq.length; }
      if (at < 0) return;
      ranges.push([n.map[at], n.map[at + len - 1] + 1]);
    });
    ranges.sort(function (a, b) { return a[0] - b[0]; });
    var merged = [];
    ranges.forEach(function (r) {
      var last = merged[merged.length - 1];
      if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
      else merged.push([r[0], r[1]]);
    });
    return merged;
  }

  /** Nodes for text[start:end] with the given ranges wrapped in <mark>. */
  function highlighted(text, ranges, start, end) {
    var nodes = [], pos = start;
    ranges.forEach(function (r) {
      var a = Math.max(r[0], start), b = Math.min(r[1], end);
      if (b <= a) return;
      if (a > pos) nodes.push(text.slice(pos, a));
      nodes.push(el("mark", { text: text.slice(a, b) }));
      pos = b;
    });
    if (pos < end) nodes.push(text.slice(pos, end));
    return nodes;
  }

  // ------------------------------------------------------------------ API

  function AskError(kind, title, message) {
    this.kind = kind; this.title = title; this.message = message;
  }

  function errorFor(status, detail, retryAfter) {
    var wait = retryAfter && /^\d+$/.test(retryAfter) ? " Try again in about " + retryAfter + " seconds." : "";
    if (status === 503 || status === 502 || status === 504) {
      return new AskError(status, "Service waking up or unavailable",
        "The free-tier server sleeps when idle and can take up to a minute to start, or a model " +
        "provider is temporarily down. Please try again shortly." + wait);
    }
    if (status === 429) {
      return new AskError(status, "Rate limit reached",
        "The free-tier model quota is used up for the moment." + (wait || " Please wait a minute and try again."));
    }
    if (status === 400 || status === 422) {
      return new AskError(status, "Request not accepted", detail || "Check the question and the date, then try again.");
    }
    if (status === 404 || status === 405) {
      return new AskError(status, "Question API not available", "This deployment does not serve " + endpoint + " yet.");
    }
    return new AskError(status, "Server error (" + status + ")",
      (detail ? detail + " " : "") + "The server could not answer this question. Please try again later.");
  }

  function detailText(body) {
    if (!body) return "";
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map(function (d) { return d && d.msg ? String(d.msg) : ""; }).filter(Boolean).join("; ");
    }
    return "";
  }

  function callAsk(payload, controller) {
    var timedOut = false;
    var timer = setTimeout(function () { timedOut = true; controller.abort(); }, TIMEOUT_MS);
    return fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    }).then(function (resp) {
      if (!resp.ok) {
        return resp.json().catch(function () { return null; }).then(function (body) {
          throw errorFor(resp.status, detailText(body), resp.headers.get("Retry-After"));
        });
      }
      return resp.json().then(function (data) {
        if (!data || typeof data !== "object" || !data.answer || !Array.isArray(data.hits)) {
          throw new AskError("shape", "Unexpected response", "The server returned a response this page does not understand.");
        }
        return data;
      }, function () {
        throw new AskError("shape", "Unexpected response", "The server did not return JSON.");
      });
    }, function (err) {
      if (err instanceof AskError) throw err;
      if (controller.signal.aborted) {
        if (timedOut) {
          throw new AskError("timeout", "Timed out", "No answer after " + TIMEOUT_MS / 1000 + " seconds. The server may be busy or waking up; please try again.");
        }
        throw new AskError("aborted", "Cancelled", "");
      }
      throw new AskError("network", "Could not reach the server", "Check your connection and try again.");
    }).finally(function () { clearTimeout(timer); });
  }

  // ------------------------------------------------------------------ rendering

  function citedSourceIndexes(result) {
    var seen = {}, out = [];
    (result.answer.citations || []).forEach(function (c) {
      var i = sourceIndexForCitation(result, c);
      if (i && !seen[i]) { seen[i] = true; out.push(i); }
    });
    return out;
  }

  /** 1-based hit index a citation points at (checks chunk_id, falls back to the index). */
  function sourceIndexForCitation(result, c) {
    var hits = result.hits;
    var s = Number(c.source);
    if (s >= 1 && s <= hits.length && hits[s - 1].chunk_id === c.chunk_id) return s;
    for (var i = 0; i < hits.length; i++) if (hits[i].chunk_id === c.chunk_id) return i + 1;
    return s >= 1 && s <= hits.length ? s : 0;
  }

  function sourceId(mode, n) { return "src-" + mode + "-" + n; }

  function jumpToSource(mode, n) {
    var card = document.getElementById(sourceId(mode, n));
    if (!card) return;
    var details = card.closest("details");
    if (details && !details.open) details.open = true;
    card.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    card.focus({ preventScroll: true });
    card.classList.remove("flash");
    void card.offsetWidth; // restart the highlight
    card.classList.add("flash");
    setTimeout(function () { card.classList.remove("flash"); }, 2200);
  }

  var MARKER_RE = /\[((?:S\s*\d+\s*[,;]?\s*)+)\]/gi;

  /** Text with [S1] / [S1, S3] markers turned into links to the source cards. */
  function inlineNodes(text, mode, nHits) {
    var nodes = [], last = 0, m;
    text = text.replace(/\*\*(.+?)\*\*/g, "$1");
    MARKER_RE.lastIndex = 0;
    while ((m = MARKER_RE.exec(text)) !== null) {
      if (m.index > last) nodes.push(text.slice(last, m.index));
      var nums = m[1].match(/\d+/g) || [];
      nodes.push("[");
      nums.forEach(function (num, i) {
        var n = parseInt(num, 10);
        if (i) nodes.push(", ");
        if (n >= 1 && n <= nHits) {
          nodes.push(el("a", {
            class: "cite", href: "#" + sourceId(mode, n), text: "S" + n,
            "aria-label": "Source " + n,
            onclick: function (ev) { ev.preventDefault(); jumpToSource(mode, n); }
          }));
        } else {
          nodes.push(el("span", { class: "cite-missing", text: "S" + n, title: "No such source" }));
        }
      });
      nodes.push("]");
      last = MARKER_RE.lastIndex;
    }
    if (last < text.length) nodes.push(text.slice(last));
    return nodes;
  }

  /** Paragraphs and simple bullet lists from plain model text. */
  function answerBody(text, mode, nHits) {
    var frag = document.createDocumentFragment();
    String(text || "").trim().split(/\n\s*\n/).forEach(function (block) {
      var lines = block.split("\n").map(function (l) { return l.trim(); }).filter(Boolean);
      if (!lines.length) return;
      var bullet = /^([-*•]|\d+[.)])\s+/;
      if (lines.every(function (l) { return bullet.test(l); })) {
        var ordered = /^\d/.test(lines[0]);
        var list = el(ordered ? "ol" : "ul");
        lines.forEach(function (l) { list.appendChild(el("li", null, inlineNodes(l.replace(bullet, ""), mode, nHits))); });
        frag.appendChild(list);
      } else {
        var p = el("p");
        lines.forEach(function (l, i) {
          if (i) p.appendChild(el("br"));
          append(p, inlineNodes(l.replace(/^#+\s*/, ""), mode, nHits));
        });
        frag.appendChild(p);
      }
    });
    return frag;
  }

  function safeUrl(url) {
    if (!url) return null;
    try {
      var u = new URL(url, window.location.href);
      return (u.protocol === "https:" || u.protocol === "http:") ? u.href : null;
    } catch (e) { return null; }
  }

  function excerptNodes(text, ranges, full) {
    var LIMIT = 700;
    if (full || text.length <= LIMIT) return highlighted(text, ranges, 0, text.length);
    var start = 0, end = LIMIT;
    if (ranges.length) {
      var a = ranges[0][0], b = ranges[ranges.length - 1][1];
      start = Math.max(0, a - 200);
      end = Math.min(text.length, Math.max(b + 200, start + LIMIT));
      if (end - start > 1400) end = Math.min(text.length, b + 100);
      // snap to word boundaries
      if (start > 0) { var sp = text.indexOf(" ", start); if (sp > 0 && sp < a) start = sp + 1; }
    }
    if (end < text.length) { var ep = text.lastIndexOf(" ", end); if (ep > start + 100) end = ep; }
    var nodes = highlighted(text, ranges, start, end);
    if (start > 0) nodes.unshift("… ");
    if (end < text.length) nodes.push(" …");
    return nodes;
  }

  function sourceCard(result, n, refDate, citationsByIndex) {
    var hit = result.hits[n - 1];
    var mode = result.mode;
    var v = validity(hit, refDate);
    var cits = citationsByIndex[n] || [];
    var verifiedQuotes = cits.filter(function (c) { return c.verified; }).map(function (c) { return c.quote; });
    var unverified = cits.filter(function (c) { return !c.verified && c.quote; });
    var text = String(hit.text || "");
    var ranges = findQuotes(text, verifiedQuotes);

    var metaBits = [];
    if (hit.issue_date) metaBits.push("Issued " + fmtDate(hit.issue_date));
    if (hit.section_ref) metaBits.push(String(hit.section_ref));
    if (hit.page) metaBits.push("p. " + hit.page);
    if (v.range) metaBits.push(v.range);

    var excerpt = el("blockquote", { class: "excerpt" });
    append(excerpt, excerptNodes(text, ranges, false));

    var actions = el("div", { class: "source-actions" });
    if (text.length > 700) {
      var expanded = false;
      var toggle = el("button", { type: "button", class: "btn btn-ghost", "aria-expanded": "false", text: "Show full passage" });
      toggle.addEventListener("click", function () {
        expanded = !expanded;
        clear(excerpt);
        append(excerpt, excerptNodes(text, ranges, expanded));
        toggle.textContent = expanded ? "Show less" : "Show full passage";
        toggle.setAttribute("aria-expanded", String(expanded));
      });
      actions.appendChild(toggle);
    }
    var href = safeUrl(hit.url);
    if (href) {
      actions.appendChild(el("a", { href: href, target: "_blank", rel: "noopener noreferrer" },
        "Open source PDF", el("span", { class: "visually-hidden", text: " (opens in a new tab)" })));
    }

    var notes = [];
    if (verifiedQuotes.length) {
      notes.push(el("p", { class: "quote-note ok", text: ranges.length
        ? "✓ Quoted text verified in this source (highlighted)."
        : "✓ Quote verified by the server." }));
    }
    unverified.forEach(function (c) {
      notes.push(el("p", { class: "quote-note bad" }, "⚠ Claimed quote not found in this source: “",
        String(c.quote).slice(0, 300), "”"));
    });

    return el("article", {
      class: "source" + (v.stale ? " is-stale" : ""), id: sourceId(mode, n), tabindex: "-1",
      "aria-label": "Source " + n + (hit.number ? ", " + hit.number : "")
    },
      el("div", { class: "source-head" },
        el("span", { class: "sid", text: "S" + n }),
        el("span", { class: "kind", text: KIND_LABELS[hit.kind] || String(hit.kind || "") }),
        el("h5", { class: "docno", text: hit.number || hit.title || "Untitled document" }),
        el("span", { class: "badge " + v.cls, text: v.label })
      ),
      metaBits.length ? el("p", { class: "source-meta", text: metaBits.join(" · ") }) : null,
      hit.number && hit.title ? el("p", { class: "source-title", text: hit.title }) : null,
      excerpt,
      notes,
      actions
    );
  }

  function planBox(result, refDate) {
    var plan = result.plan || {};
    var rows = [];
    var asOf = plan.as_of || (result.answer && result.answer.as_of);
    var how = plan.as_of_explicit ? "taken from your question or date field" : "defaulted to today";
    rows.push(["As of", [fmtDate(asOf), " ", el("span", { class: "hint", text: "(" + how + ")" })]]);
    if (refDate && asOf && refDate !== asOf) {
      rows.push(["Badges use", fmtDate(refDate)]);
    }
    if (plan.question_type) rows.push(["Question type", String(plan.question_type).replace(/_/g, " ")]);
    if (plan.refs && plan.refs.length) {
      rows.push(["References", el("span", { class: "tags" }, plan.refs.map(function (r) { return el("span", { class: "tag", text: r }); }))]);
    }
    if (plan.search_queries && plan.search_queries.length) {
      rows.push(["Search queries", el("ul", null, plan.search_queries.map(function (q) { return el("li", { text: q }); }))]);
    }
    var dl = el("dl", { class: "plan" });
    rows.forEach(function (r) { dl.appendChild(el("dt", { text: r[0] })); dl.appendChild(el("dd", null, r[1])); });

    var timings = result.timings_ms || {};
    var keys = Object.keys(timings).filter(function (k) { return typeof timings[k] === "number"; });
    var total = typeof timings.total === "number" ? timings.total : keys.reduce(function (s, k) { return s + timings[k]; }, 0);
    var steps = keys.filter(function (k) { return k !== "total"; });
    var timingList = null;
    if (steps.length) {
      var max = Math.max.apply(null, steps.map(function (k) { return timings[k]; }).concat([1]));
      timingList = el("ul", { class: "timings", "aria-label": "Time per step" }, steps.map(function (k) {
        var bar = el("span");
        bar.style.width = Math.max(1, Math.round(timings[k] / max * 100)) + "%";
        return el("li", null,
          el("span", { class: "step", text: k.replace(/_/g, " "), title: k }),
          el("span", { class: "bar", "aria-hidden": "true" }, bar),
          el("span", { class: "ms", text: fmtMs(timings[k]) }));
      }));
    }
    return el("details", { class: "plan-box" },
      el("summary", { text: "How it was answered" + (total ? " · " + fmtMs(total) : "") }),
      dl,
      timingList ? el("p", { class: "section-title", text: "Timings" }) : null,
      timingList);
  }

  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms"; }

  function renderResult(column, result, refDate) {
    var mode = result.mode in MODES ? result.mode : column.getAttribute("data-mode");
    result.mode = mode;
    var body = column.querySelector(".column-body");
    clear(body);
    column.removeAttribute("aria-busy");

    var answer = result.answer || {};
    var hits = result.hits || [];
    var cited = citedSourceIndexes(result);
    var citationsByIndex = {};
    (answer.citations || []).forEach(function (c) {
      var i = sourceIndexForCitation(result, c);
      if (i) (citationsByIndex[i] = citationsByIndex[i] || []).push(c);
    });
    var stale = cited.filter(function (i) { return validity(hits[i - 1], refDate).stale; });
    var verified = (answer.citations || []).filter(function (c) { return c.verified; }).length;

    var chips = el("ul", { class: "summary-chips", "aria-label": "Answer summary" });
    if (answer.abstained) chips.appendChild(el("li", { class: "badge warn", text: "Declined to answer" }));
    chips.appendChild(el("li", { class: "badge info", text: cited.length + " source" + (cited.length === 1 ? "" : "s") + " cited" }));
    if (answer.citations && answer.citations.length) {
      chips.appendChild(el("li", { class: "badge " + (verified === answer.citations.length ? "ok" : "warn"),
        text: verified + "/" + answer.citations.length + " quotes verified" }));
    }
    if (cited.length && refDate) {
      chips.appendChild(stale.length
        ? el("li", { class: "badge bad", text: stale.length + " cited not in force on " + fmtDate(refDate) })
        : el("li", { class: "badge ok", text: "All cited in force on " + fmtDate(refDate) }));
    }
    body.appendChild(chips);

    var ans = el("div", { class: "answer" + (answer.abstained ? " abstained" : "") });
    if (answer.abstained) {
      ans.appendChild(el("p", { class: "abstain-title", text: "Not enough evidence to answer" }));
    }
    ans.appendChild(answerBody(answer.text || (answer.abstained ? "No answer: the sources in force on this date do not support one." : ""), mode, hits.length));
    body.appendChild(ans);

    if (hits.length) {
      var citedList = el("div", { class: "sources" });
      cited.forEach(function (n) { citedList.appendChild(sourceCard(result, n, refDate, citationsByIndex)); });
      if (cited.length) {
        body.appendChild(el("h4", { class: "section-title", text: "Cited sources" }));
        body.appendChild(citedList);
      }
      var others = [];
      for (var n = 1; n <= hits.length; n++) if (cited.indexOf(n) < 0) others.push(n);
      if (others.length) {
        var otherList = el("div", { class: "sources" });
        others.forEach(function (k) { otherList.appendChild(sourceCard(result, k, refDate, citationsByIndex)); });
        var label = cited.length ? "Other retrieved passages (" + others.length + ")" : "Retrieved passages (" + others.length + ")";
        body.appendChild(el("details", { class: "more", open: cited.length ? null : true },
          el("summary", { text: label }), otherList));
      }
    } else {
      body.appendChild(el("p", { class: "hint", text: "No passages were retrieved." }));
    }
    body.appendChild(planBox(result, refDate));
    return { stale: stale.length, cited: cited.length };
  }

  function renderError(column, err, retry) {
    var body = column.querySelector(".column-body");
    clear(body);
    column.removeAttribute("aria-busy");
    body.appendChild(el("div", { class: "error-box", role: "alert" },
      el("strong", { text: err.title || "Something went wrong" }),
      err.message ? el("p", { text: err.message }) : null,
      el("button", { type: "button", class: "btn btn-ghost", text: "Try again", onclick: retry })));
  }

  function makeColumn(mode) {
    var info = MODES[mode];
    var started = Date.now();
    var elapsed = el("span", { text: "" });
    var slow = el("p", { class: "hint", hidden: true,
      text: "Still working. The first request after a quiet period can take up to a minute while the free-tier server wakes up." });
    var loading = el("div", { class: "loading" }, el("span", { class: "spinner", "aria-hidden": "true" }),
      el("span", null, "Retrieving sources and drafting an answer… ", elapsed));
    var column = el("section", { class: "column mode-" + mode, "data-mode": mode, "aria-busy": "true",
      "aria-labelledby": "col-title-" + mode },
      el("div", { class: "column-head" },
        el("h3", { id: "col-title-" + mode, text: info.label }),
        el("p", { class: "hint", text: info.sub })),
      el("div", { class: "column-body" }, loading, slow,
        el("div", { class: "skeleton" }), el("div", { class: "skeleton" }), el("div", { class: "skeleton" })));
    var tick = setInterval(function () {
      if (!column.hasAttribute("aria-busy")) { clearInterval(tick); return; }
      var s = Math.round((Date.now() - started) / 1000);
      elapsed.textContent = s ? "(" + s + " s)" : "";
      if (Date.now() - started > SLOW_HINT_MS) slow.hidden = false;
    }, 1000);
    return column;
  }

  function compareBanner(state) {
    var old = resultsEl.querySelector(".compare-banner");
    if (old) old.remove();
    var a = state.summary.asof, b = state.summary.baseline;
    if (!a || !b || !state.refDate) return;
    var date = fmtDate(state.refDate);
    var msg;
    if (b.stale > a.stale) {
      msg = "On " + date + ", the baseline cited " + b.stale + " source" + (b.stale === 1 ? "" : "s") +
        " that " + (b.stale === 1 ? "was" : "were") + " not in force; AsOf cited " + (a.stale || "none") + ".";
    } else if (b.stale || a.stale) {
      msg = "Cited sources not in force on " + date + ": AsOf " + a.stale + ", baseline " + b.stale + ".";
    } else {
      msg = "Both systems cited only sources in force on " + date + ". Compare the answers and sources below.";
    }
    var banner = el("div", { class: "compare-banner" + (b.stale || a.stale ? " has-stale" : ""), role: "note" },
      el("strong", { text: "Comparison: " }), msg);
    resultsEl.insertBefore(banner, resultsEl.firstChild);
  }

  // ------------------------------------------------------------------ flow

  function syncUrl(question, asOf, compare) {
    try {
      var params = new URLSearchParams();
      params.set("q", question);
      if (asOf) params.set("as_of", asOf);
      if (compare) params.set("compare", "1");
      history.replaceState(null, "", window.location.pathname + "?" + params.toString() + "#ask");
    } catch (e) { /* non-essential */ }
  }

  function setBusy(busy) {
    btn.setAttribute("aria-busy", busy ? "true" : "false");
    btn.textContent = busy ? "Asking…" : "Ask";
  }

  function submit() {
    var question = qEl.value.trim();
    var asOf = dateEl.value;
    qEl.removeAttribute("aria-invalid");
    dateEl.removeAttribute("aria-invalid");
    if (question.length < 3) {
      qEl.setAttribute("aria-invalid", "true");
      setStatus("Please type a question first.", true);
      qEl.focus();
      return;
    }
    if (asOf && (!isISODate(asOf) || asOf < MIN_DATE)) {
      dateEl.setAttribute("aria-invalid", "true");
      setStatus("Pick a date on or after 1 July 2017, when GST began, or leave it empty for today.", true);
      dateEl.focus();
      return;
    }
    if (run) run.controllers.forEach(function (c) { c.abort(); });

    var compare = cmpEl.checked;
    var modes = compare ? ["asof", "baseline"] : ["asof"];
    var state = { id: ++runCounter, controllers: [], results: {}, summary: {}, userDate: asOf || null,
                  refDate: asOf || null, pending: modes.length, columns: {} };
    run = state;
    syncUrl(question, asOf, compare);

    clear(resultsEl);
    resultsEl.classList.toggle("is-compare", compare);
    modes.forEach(function (mode) {
      var column = makeColumn(mode);
      state.columns[mode] = column;
      resultsEl.appendChild(column);
    });
    setBusy(true);
    setStatus(compare ? "Asking AsOf and the baseline in parallel…" : "Asking…");

    modes.forEach(function (mode) { start(state, mode, question, asOf); });
  }

  function start(state, mode, question, asOf) {
    var controller = new AbortController();
    state.controllers.push(controller);
    var payload = { question: question, mode: mode };
    if (asOf) payload.as_of = asOf;
    callAsk(payload, controller).then(function (result) {
      if (run !== state) return;
      state.results[mode] = result;
      // Badges for both columns use the date AsOf interpreted (or the date the user picked).
      if (!state.userDate) {
        var ra = state.results.asof;
        var d = ra && ra.plan && ra.plan.as_of;
        if (!d && mode === "baseline" && result.plan) d = result.plan.as_of;
        if (isISODate(d) && d !== state.refDate) {
          state.refDate = d;
          if (mode === "asof" && state.results.baseline) {
            state.summary.baseline = renderResult(state.columns.baseline, state.results.baseline, d);
          }
        }
      }
      state.summary[mode] = renderResult(state.columns[mode], result, state.refDate || todayISO());
    }, function (err) {
      if (run !== state || err.kind === "aborted") return;
      state.summary[mode] = null;
      renderError(state.columns[mode], err, function () {
        state.columns[mode].setAttribute("aria-busy", "true");
        var fresh = makeColumn(mode);
        resultsEl.replaceChild(fresh, state.columns[mode]);
        state.columns[mode] = fresh;
        state.pending += 1;
        setBusy(true);
        setStatus("Retrying…");
        start(state, mode, question, asOf);
      });
    }).finally(function () {
      if (run !== state) return;
      state.pending -= 1;
      if (state.pending > 0) return;
      setBusy(false);
      var failed = Object.keys(state.columns).filter(function (m) { return !state.results[m]; });
      if (Object.keys(state.columns).length > 1) compareBanner(state);
      setStatus(failed.length === Object.keys(state.columns).length ? "The request failed; see the message below."
        : failed.length ? "One of the two requests failed; see below." : "Answer ready.", !!failed.length);
    });
  }

  // ------------------------------------------------------------------ wiring

  form.addEventListener("submit", function (ev) { ev.preventDefault(); submit(); });
  qEl.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); submit(); }
  });
  Array.prototype.forEach.call(form.querySelectorAll(".chip[data-q]"), function (chip) {
    chip.addEventListener("click", function () {
      qEl.value = chip.getAttribute("data-q") || "";
      if (chip.hasAttribute("data-as-of")) dateEl.value = chip.getAttribute("data-as-of");
      submit();
    });
  });

  // Prefill from a shared link: /?q=...&as_of=YYYY-MM-DD&compare=1 (does not auto-submit).
  try {
    var params = new URLSearchParams(window.location.search);
    if (params.get("q")) qEl.value = params.get("q").slice(0, 1000);
    if (isISODate(params.get("as_of"))) dateEl.value = params.get("as_of");
    if (params.get("compare") === "1") cmpEl.checked = true;
  } catch (e) { /* ignore */ }
  dateEl.setAttribute("max", "9999-12-31");
})();
