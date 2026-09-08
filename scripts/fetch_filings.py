"""Download the most recent 10-K filing for each ticker in COMPANIES from SEC EDGAR.

Usage:
    uv run scripts/fetch_filings.py

SEC EDGAR requires a User-Agent header identifying the requester (fair-access
policy: https://www.sec.gov/os/webmaster-faq#developers).
"""

import time
from pathlib import Path

import requests

COMPANIES = ["AAPL", "JPM", "XOM", "JNJ", "WMT", "TSLA"]
EDGAR_CONTACT = "Dan Ukendi dan.ukendi1@gmail.com"

# SEC's ticker->CIK map currently points XOM at "ExxonMobil Holdings Corp"
# (CIK 2115436), a newer shell entity with no 10-K filed yet. The actual
# operating company that files 10-Ks is still "EXXON MOBIL CORP" (CIK
# 34088), so we override the lookup for this one ticker.
CIK_OVERRIDES = {"XOM": "34088"}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{document}"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": EDGAR_CONTACT})
    return session


def build_ticker_to_cik(session: requests.Session) -> dict[str, str]:
    resp = session.get(TICKERS_URL, timeout=30)
    resp.raise_for_status()
    rows = resp.json().values()
    return {row["ticker"].upper(): str(row["cik_str"]) for row in rows}


def latest_10k(session: requests.Session, cik: str) -> dict:
    resp = session.get(SUBMISSIONS_URL.format(cik=cik), timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession_no": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
    raise ValueError(f"No 10-K found for CIK {cik}")


def download_filing(session: requests.Session, ticker: str, cik: str, filing: dict) -> Path:
    accession_no_dashes = filing["accession_no"].replace("-", "")
    url = ARCHIVE_URL.format(
        cik=int(cik),
        accession_no_dashes=accession_no_dashes,
        document=filing["primary_document"],
    )
    resp = session.get(url, timeout=60)
    resp.raise_for_status()

    suffix = Path(filing["primary_document"]).suffix or ".htm"
    out_path = DATA_DIR / f"{ticker}_{filing['filing_date']}_10K{suffix}"
    out_path.write_bytes(resp.content)
    return out_path


def main() -> None:
    session = get_session()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching ticker -> CIK map...")
    ticker_to_cik = build_ticker_to_cik(session)

    for ticker in COMPANIES:
        cik = CIK_OVERRIDES.get(ticker, ticker_to_cik.get(ticker))
        if cik is None:
            print(f"[skip] {ticker}: not found in EDGAR ticker map")
            continue

        filing = latest_10k(session, cik)
        out_path = download_filing(session, ticker, cik, filing)
        print(f"[ok] {ticker}: {filing['filing_date']} -> {out_path.relative_to(DATA_DIR.parent.parent)}")

        time.sleep(0.2)  # stay well under EDGAR's rate limit


if __name__ == "__main__":
    main()
