"""MCP integrations and free public APIs for the travel planner.

The module keeps external integrations in one place. Optional local MCP
servers can be enabled through environment variables, while public APIs
provide the fallback data used by the agents.
"""

import asyncio

import csv

import io

import math

import os

import re

import time

from datetime import date

from pathlib import Path

from typing import Any

from urllib.parse import quote

import httpx

from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient

# ============================================================

# ENVIRONMENT

# ============================================================

load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ============================================================

# MCP CLIENT

# ============================================================

# Local MCP servers are optional.

# Do NOT hard-code Windows paths here.

# For local MCP usage, set these environment variables in .env.

# On Streamlit Cloud/VPS, leave them empty so local stdio servers are skipped.

_AVIATION_PY = os.getenv("AVIATION_MCP_PYTHON", "").strip()

_WEATHER_PY = os.getenv("WEATHER_MCP_PYTHON", "").strip()

_WEATHER_SCRIPT = os.getenv("WEATHER_MCP_SCRIPT", "").strip()

_servers: dict = {

    "tavily": {

        "transport": "streamable_http",

        "url": (

            "https://mcp.tavily.com/mcp/"

            f"?tavilyApiKey={TAVILY_API_KEY}"

        ),

    },

}

if _AVIATION_PY and os.path.isfile(_AVIATION_PY):

    _servers["aviationstack"] = {

        "transport": "stdio",

        "command": _AVIATION_PY,

        "args": ["-m", "aviationstack_mcp", "mcp", "run"],

        "env": {"AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY or ""},

    }

else:

    print("AviationStack local MCP not configured: skipped.")

if (

    _WEATHER_PY

    and _WEATHER_SCRIPT

    and os.path.isfile(_WEATHER_PY)

    and os.path.isfile(_WEATHER_SCRIPT)

):

    _servers["weather"] = {

        "transport": "stdio",

        "command": _WEATHER_PY,

        "args": [_WEATHER_SCRIPT],

        "env": {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY or ""},

    }

else:

    print("Custom Weather local MCP not configured: skipped.")

client = MultiServerMCPClient(_servers)

TOOL_SERVER = {

    "tavily_search": "tavily",

    "list_airports": "aviationstack",

    "list_airlines": "aviationstack",

    "get_current_weather": "weather",

    "get_forecast": "weather",

}

_tools_cache: dict[str, list] = {}

async def get_tools(server_name: str):

    if server_name in _tools_cache:

        return _tools_cache[server_name]

    try:

        tools = await client.get_tools(server_name=server_name)

    except Exception as e:

        # Never print the Tavily URL: it contains the API key.

        print(f"\n========== MCP ERROR ({server_name}) ==========")

        print(type(e).__name__)

        print(str(e).replace(TAVILY_API_KEY or "<none>", "***"))

        print("===============================================\n")

        raise

    _tools_cache[server_name] = tools

    return tools

async def call_tool(tool_name: str, args: dict | None = None):

    server_name = TOOL_SERVER.get(tool_name)

    if server_name is None:

        raise ValueError(f"Unknown tool '{tool_name}'")

    tools = await get_tools(server_name)

    tool = next((t for t in tools if t.name == tool_name), None)

    if tool is None:

        available = [t.name for t in tools]

        raise ValueError(

            f"Tool '{tool_name}' not found on server "

            f"'{server_name}'. Available tools: {available}"

        )

    return await tool.ainvoke(args or {})

# MCP wrappers ----------------

async def tavily_search(query: str):

    return await call_tool("tavily_search", {"query": query})

async def list_airports(search: str = "", limit: int = 10):

    return await call_tool(

        "list_airports",

        {"search": search, "limit": limit, "offset": 0},

    )

async def list_airlines(search: str = "", limit: int = 10):

    return await call_tool(

        "list_airlines",

        {"search": search, "limit": limit, "offset": 0},

    )

async def current_weather(city: str):

    return await call_tool("get_current_weather", {"city": city})

async def forecast(city: str):

    return await call_tool("get_forecast", {"city": city})

# ============================================================

# Shared HTTP and cache helpers

# ============================================================

HTTP_HEADERS = {

    # Overpass / Nominatim-style services ask for an identifying User-Agent.

    "User-Agent": "MultiAgentTravelPlanner/1.0 (educational project)",

    "Accept": "application/json",

}

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

COUNTRIES_URL = "https://restcountries.com/v3.1"

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"

ER_API_URL = "https://open.er-api.com/v6/latest"

NAGER_URL = "https://date.nager.at/api/v3/PublicHolidays"

OVERPASS_ENDPOINTS = [

    "https://overpass-api.de/api/interpreter",

    "https://overpass.kumi.systems/api/interpreter",

]

AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# Open-Meteo gives ~16 forecast days; stay a little inside the limit

# (the location's "today" can differ from ours by a day).

FORECAST_HORIZON_DAYS = 14

_CACHE: dict[Any, tuple[float, Any]] = {}

def _cache_get(key):

    item = _CACHE.get(key)

    if item and item[0] > time.time():

        return True, item[1]

    return False, None

def _cache_set(key, value, ttl: float):

    _CACHE[key] = (time.time() + ttl, value)

async def _http_json(

    url: str,

    params: dict | None = None,

    *,

    method: str = "GET",

    data: dict | None = None,

    timeout: float = 20.0,

):

    """

    GET/POST JSON with 3 attempts on timeouts, connection errors and

    429/502/503/504 (public free APIs are sometimes slow or busy).

    """

    last_exc: Exception | None = None

    for attempt in range(3):

        try:

            async with httpx.AsyncClient(

                timeout=timeout,

                headers=HTTP_HEADERS,

                follow_redirects=True,

            ) as http:

                if method == "POST":

                    resp = await http.post(url, data=data)

                else:

                    resp = await http.get(url, params=params)

            if resp.status_code == 204:

                return None

            if resp.status_code in (429, 502, 503, 504) and attempt < 2:

                last_exc = httpx.HTTPStatusError(

                    f"HTTP {resp.status_code}",

                    request=resp.request,

                    response=resp,

                )

                await asyncio.sleep(1.5 * (attempt + 1))

                continue

            resp.raise_for_status()

            return resp.json()

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:

            last_exc = exc

            if attempt < 2:

                await asyncio.sleep(1.5 * (attempt + 1))

                continue

            raise

    raise last_exc or RuntimeError("HTTP request failed")

def haversine_km(lat1, lon1, lat2, lon2) -> float:

    """Great-circle distance in km."""

    r = 6371.0088

    p1, p2 = math.radians(lat1), math.radians(lat2)

    dphi = p2 - p1

    dlmb = math.radians(lon2 - lon1)

    a = (

        math.sin(dphi / 2) ** 2

        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2

    )

    return 2 * r * math.asin(math.sqrt(a))

def _shift_year(d: date, delta: int) -> date:

    try:

        return d.replace(year=d.year + delta)

    except ValueError:  # 29 Feb

        return d.replace(year=d.year + delta, day=28)

# ============================================================

# Geocoding and country helpers

# ============================================================

def _geo_from_openmeteo(r: dict, exact: bool) -> dict:

    code = str(r.get("feature_code", ""))

    return {

        "name": r.get("name", ""),

        "lat": r.get("latitude"),

        "lon": r.get("longitude"),

        "country": r.get("country", ""),

        "country_code": (r.get("country_code") or "").upper(),

        "admin1": r.get("admin1", ""),

        "timezone": r.get("timezone", ""),

        "population": r.get("population") or 0,

        "feature_code": code,

        "exact": exact,

        "region_level": code.startswith("ADM"),

        "source": "Open-Meteo",

    }

def _geo_score(r: dict, prefer: str) -> float:
    """Rank candidates by population, with a country preference bonus."""
    score = float((r.get("population") or 0) + 1)
    if prefer and (r.get("country_code") or "").upper() == prefer:
        score *= 3
    return score

_REGION_TYPES = {

    "state", "region", "province", "county", "state_district", "island",

}

async def _nominatim(query: str, cc_filter: str, prefer: str) -> dict | None:

    """

    OpenStreetMap Nominatim (free, 1 request/second policy, cached).

    Handles states and regions such as "Goa" or "Kerala" that the

    city-oriented Open-Meteo geocoder does not know.

    """

    params = {

        "q": query,

        "format": "jsonv2",

        "limit": 6,

        "addressdetails": 1,

        "accept-language": "en",

    }

    if cc_filter:

        params["countrycodes"] = cc_filter.lower()

    data = await _http_json(NOMINATIM_URL, params)

    if not data:

        return None

    # keep places/boundaries (ignore shops or streets that merely share the name)

    places = [d for d in data if d.get("category") in ("boundary", "place")] or data

    def score(item: dict) -> float:

        importance = float(item.get("importance") or 0.01)

        cc = ((item.get("address") or {}).get("country_code") or "").upper()

        return importance * (2.0 if prefer and cc == prefer else 1.0)

    best = max(places, key=score)

    addr = best.get("address") or {}

    name = best.get("name") or str(best.get("display_name", "")).split(",")[0].strip()

    region = best.get("addresstype") in _REGION_TYPES

    try:

        lat, lon = float(best["lat"]), float(best["lon"])

    except (KeyError, ValueError, TypeError):

        return None

    return {

        "name": name,

        "lat": lat,

        "lon": lon,

        "country": addr.get("country", ""),

        "country_code": (addr.get("country_code") or "").upper(),

        "admin1": addr.get("state") or addr.get("region") or "",

        "timezone": "",

        "population": 0,

        "feature_code": "ADM1" if region else "PPL",

        "exact": name.strip().lower() == query.strip().lower(),

        "region_level": region,

        "source": "OpenStreetMap Nominatim",

    }

async def geocode(

    name: str,

    country_code: str | None = None,

    prefer_country: str | None = None,

) -> dict | None:

    """

    Place -> coordinates.

    1. Open-Meteo geocoder: only EXACT name matches are trusted; among

       them the most populated wins, boosted if in prefer_country

       (e.g. the origin's country, so "Hyderabad" from India is India).

    2. No exact match (states/regions like "Goa"): OpenStreetMap Nominatim.

    3. Last resort: the fuzzy Open-Meteo result, flagged exact=False so

       the UI can warn the user.

    "Goa, India" works too: the part after the comma is a country hint.

    """

    name = (name or "").strip()

    if not name:

        return None

    parts = [p.strip() for p in name.split(",") if p.strip()]

    base = parts[0] if parts else name

    cc_filter = (country_code or "").upper()

    if not cc_filter and len(parts) > 1:

        hint = await country_by_name(parts[-1])

        if hint and hint.get("cca2"):

            cc_filter = hint["cca2"].upper()

    prefer = (prefer_country or "").upper()

    key = ("geocode3", base.lower(), cc_filter, prefer)

    hit, value = _cache_get(key)

    if hit:

        return value

    candidates: list[str] = []

    for cand in (base, re.sub(r"^the\s+", "", base, flags=re.IGNORECASE)):

        if cand and cand not in candidates:

            candidates.append(cand)

    had_error = False

    result = None

    fuzzy = None

    for cand in candidates:

        params = {

            "name": cand,

            "count": 10,

            "language": "en",

            "format": "json",

        }

        if cc_filter:

            params["countryCode"] = cc_filter

        try:

            data = await _http_json(GEOCODING_URL, params)

        except Exception as exc:

            had_error = True

            print("Geocoding error:", repr(exc))

            continue

        results = (data or {}).get("results") or []

        exact = [

            r for r in results

            if str(r.get("name", "")).strip().lower() == cand.lower()

        ]

        if exact:
            # Prefer an exact match in the preferred country when one exists.
            preferred = [
                r for r in exact
                if prefer
                and (r.get("country_code") or "").upper() == prefer
            ]

            if preferred:
                best = max(preferred, key=lambda r: _geo_score(r, prefer))
                result = _geo_from_openmeteo(best, exact=True)
                break

            best = max(exact, key=lambda r: _geo_score(r, prefer))

            # A short region name can have an exact match in another country.
            # Example: Open-Meteo may return Bicol Region, Philippines for
            # "Goa". If the origin country is known, verify an ADM result
            # with Nominatim before accepting it.
            if prefer and str(best.get("feature_code", "")).startswith("ADM"):
                try:
                    preferred_region = await _nominatim(base, prefer, prefer)
                    if (
                        preferred_region
                        and preferred_region.get("country_code") == prefer
                        and preferred_region.get("region_level")
                    ):
                        result = preferred_region
                        break
                except Exception as exc:
                    had_error = True
                    print("Preferred-region lookup error:", repr(exc))

            result = _geo_from_openmeteo(best, exact=True)
            break

        if results and fuzzy is None:

            fuzzy = results[0]

    if result is None:

        try:

            result = await _nominatim(base, cc_filter, prefer)

        except Exception as exc:

            had_error = True

            print("Nominatim error:", repr(exc))

    if result is None and fuzzy is not None:

        result = _geo_from_openmeteo(fuzzy, exact=False)

    if not had_error:

        _cache_set(key, result, 24 * 3600 if result else 600)

    return result

def _normalize_country(obj: dict) -> dict:

    currencies = obj.get("currencies") or {}

    code = next(iter(currencies), "")

    cur = currencies.get(code, {}) if code else {}

    name_obj = obj.get("name")

    common = name_obj.get("common", "") if isinstance(name_obj, dict) else str(name_obj or "")

    return {

        "name": common,

        "cca2": obj.get("cca2", ""),

        "capital": (obj.get("capital") or [""])[0],

        "currency_code": code,

        "currency_name": cur.get("name", ""),

        "currency_symbol": cur.get("symbol", ""),

        "languages": sorted((obj.get("languages") or {}).values()),

        "region": obj.get("region", ""),

    }

_COUNTRY_FIELDS = "name,cca2,capital,currencies,languages,region"

def _first_country(data) -> dict | None:

    """

    REST Countries may return a list, a single object, or an error object

    like {"status": 404, "message": "Not Found"}. Return one country dict

    or None.

    """

    if isinstance(data, list):

        data = data[0] if data else None

    if isinstance(data, dict) and (data.get("cca2") or data.get("name")):

        return data

    return None

async def country_info(country_code: str) -> dict | None:

    """REST Countries by ISO alpha-2/alpha-3 code."""

    code = (country_code or "").strip().upper()

    if not code:

        return None

    key = ("country", code)

    hit, value = _cache_get(key)

    if hit:

        return value

    try:

        data = await _http_json(

            f"{COUNTRIES_URL}/alpha/{quote(code)}",

            {"fields": _COUNTRY_FIELDS},

        )

    except Exception as exc:

        print("Country info error:", repr(exc))

        return None

    country = _first_country(data)

    result = _normalize_country(country) if country else None

    _cache_set(key, result, 7 * 24 * 3600)

    return result

_COUNTRY_ALIASES = {

    "uk": "GB",

    "u.k.": "GB",

    "england": "GB",

    "scotland": "GB",

    "usa": "US",

    "u.s.a.": "US",

    "u.s.": "US",

    "america": "US",

    "uae": "AE",

    "holland": "NL",

}

async def country_by_name(name: str) -> dict | None:

    """

    Exact country-name match (so 'Japan' or 'The Netherlands' is detected

    as a COUNTRY-level destination, but 'Paris' is not).

    """

    n = (name or "").strip()

    if not n:

        return None

    alias = _COUNTRY_ALIASES.get(n.lower())

    if alias:

        return await country_info(alias)

    key = ("country_name", n.lower())

    hit, value = _cache_get(key)

    if hit:

        return value

    candidates: list[str] = []

    for cand in (n, re.sub(r"^the\s+", "", n, flags=re.IGNORECASE)):

        if cand and cand not in candidates:

            candidates.append(cand)

    result = None

    had_error = False

    for cand in candidates:

        try:

            data = await _http_json(

                f"{COUNTRIES_URL}/name/{quote(cand)}",

                {"fullText": "true", "fields": _COUNTRY_FIELDS},

            )

        except httpx.HTTPStatusError as exc:

            if exc.response.status_code != 404:

                had_error = True

                print("REST Countries HTTP error:", exc.response.status_code, cand)

            continue  # 404 = not a country name

        except Exception as exc:

            had_error = True

            print("Country lookup error:", repr(exc))

            continue

        country = _first_country(data)

        if country:

            result = _normalize_country(country)

            break

    if not had_error:

        _cache_set(key, result, 7 * 24 * 3600)

    return result

# Main tourist gateway city and currency per country. Used (a) to anchor

# COUNTRY-level destinations on a real city (not the geographic centre,

# which can be in the mountains), and (b) as a fallback when REST

# Countries is unreachable.

_COUNTRY_DEFAULTS = {

    "IN": ("Delhi", "INR"), "US": ("New York", "USD"), "GB": ("London", "GBP"),

    "FR": ("Paris", "EUR"), "DE": ("Berlin", "EUR"), "IT": ("Rome", "EUR"),

    "ES": ("Madrid", "EUR"), "PT": ("Lisbon", "EUR"), "NL": ("Amsterdam", "EUR"),

    "BE": ("Brussels", "EUR"), "CH": ("Zurich", "CHF"), "AT": ("Vienna", "EUR"),

    "GR": ("Athens", "EUR"), "IE": ("Dublin", "EUR"), "CZ": ("Prague", "CZK"),

    "PL": ("Warsaw", "PLN"), "HU": ("Budapest", "HUF"), "HR": ("Zagreb", "EUR"),

    "SE": ("Stockholm", "SEK"), "NO": ("Oslo", "NOK"), "DK": ("Copenhagen", "DKK"),

    "FI": ("Helsinki", "EUR"), "IS": ("Reykjavik", "ISK"), "TR": ("Istanbul", "TRY"),

    "AE": ("Dubai", "AED"), "SA": ("Riyadh", "SAR"), "QA": ("Doha", "QAR"),

    "EG": ("Cairo", "EGP"), "MA": ("Marrakech", "MAD"), "ZA": ("Cape Town", "ZAR"),

    "KE": ("Nairobi", "KES"), "JP": ("Tokyo", "JPY"), "KR": ("Seoul", "KRW"),

    "CN": ("Beijing", "CNY"), "HK": ("Hong Kong", "HKD"), "TW": ("Taipei", "TWD"),

    "SG": ("Singapore", "SGD"), "MY": ("Kuala Lumpur", "MYR"), "TH": ("Bangkok", "THB"),

    "VN": ("Hanoi", "VND"), "ID": ("Denpasar", "IDR"), "PH": ("Manila", "PHP"),

    "KH": ("Phnom Penh", "KHR"), "LK": ("Colombo", "LKR"), "NP": ("Kathmandu", "NPR"),

    "BD": ("Dhaka", "BDT"), "MV": ("Male", "MVR"), "BT": ("Thimphu", "BTN"),

    "AU": ("Sydney", "AUD"), "NZ": ("Auckland", "NZD"), "CA": ("Toronto", "CAD"),

    "MX": ("Mexico City", "MXN"), "BR": ("Rio de Janeiro", "BRL"),

    "AR": ("Buenos Aires", "ARS"), "PE": ("Lima", "PEN"), "CL": ("Santiago", "CLP"),

    "CO": ("Bogota", "COP"), "IL": ("Tel Aviv", "ILS"), "JO": ("Amman", "JOD"),

}

def _fallback_country(geo: dict) -> dict:

    """Minimal country info from the geocoder result (REST Countries down)."""

    cc = (geo.get("country_code") or "").upper()

    main_city, currency = _COUNTRY_DEFAULTS.get(cc, ("", ""))

    return {

        "name": geo.get("country") or geo.get("name", ""),

        "cca2": cc,

        "capital": main_city,

        "currency_code": currency,

        "currency_name": "",

        "currency_symbol": "",

        "languages": [],

        "region": "",

    }

async def resolve_place(name: str, prefer_country: str = "") -> dict | None:

    """

    Resolve a free-text place into:

    {

      query, label, anchor{lat,lon,...}, country{...}, is_country_level

    }

    Country-level destinations (e.g. "Japan") are anchored on the main

    gateway city (or the capital), so weather, hotels, attractions and

    airports are looked up for a real city, never the geographic centre.

    Country-level is detected by REST Countries OR by the geocoder

    returning a country feature, so it still works if one is unreachable.

    """

    name = (name or "").strip()

    if not name:

        return None

    country = await country_by_name(name)

    geo = None

    if not country:

        geo = await geocode(name, prefer_country=prefer_country)

        if not geo:

            return None

        if (

            str(geo.get("feature_code", "")).startswith("PCL")

            and geo.get("country_code")

        ):

            country = (

                await country_info(geo["country_code"])

                or _fallback_country(geo)

            )

    # ---------------- country-level ----------------

    if country:

        cc = (country.get("cca2") or (geo or {}).get("country_code") or "").upper()

        main_city = _COUNTRY_DEFAULTS.get(cc, ("", ""))[0] or country.get("capital")

        anchor = await geocode(main_city, cc) if main_city else None

        if anchor is None:

            anchor = geo or await geocode(name)

        if anchor is None:

            return None

        # make sure currency etc. are never empty if we know them

        if not country.get("currency_code") and cc in _COUNTRY_DEFAULTS:

            country = {**country, "currency_code": _COUNTRY_DEFAULTS[cc][1]}

        return {

            "query": name,

            "label": f"{country.get('name') or name} (anchored on {anchor.get('name', main_city)})",

            "anchor": anchor,

            "country": country,

            "is_country_level": True,

            "is_region_level": False,

            "exact": True,

        }

    # ---------------- city-level ----------------

    if geo.get("country_code"):

        country = await country_info(geo["country_code"]) or _fallback_country(geo)

    else:

        country = _fallback_country(geo)

    parts: list[str] = []

    for part in (geo.get("name"), geo.get("admin1"), geo.get("country")):

        if part and (not parts or parts[-1] != part):

            parts.append(part)

    return {

        "query": name,

        "label": ", ".join(parts),

        "anchor": geo,

        "country": country,

        "is_country_level": False,

        "is_region_level": bool(geo.get("region_level")),

        "exact": bool(geo.get("exact", True)),

    }

# ============================================================

# Exchange rates

# ============================================================

async def exchange_rate(base: str, target: str) -> dict | None:

    """

    Live-ish exchange rate (ECB daily via Frankfurter, fallback ER-API).

    Returns {"rate", "date", "source"} or None.

    """

    base = (base or "").upper()

    target = (target or "").upper()

    if not base or not target:

        return None

    if base == target:

        return {

            "rate": 1.0,

            "date": date.today().isoformat(),

            "source": "identity",

        }

    key = ("fx", base, target)

    hit, value = _cache_get(key)

    if hit:

        return value

    result = None

    try:

        data = await _http_json(

            FRANKFURTER_URL,

            {"base": base, "symbols": target},

        )

        rate = ((data or {}).get("rates") or {}).get(target)

        if rate:

            result = {

                "rate": float(rate),

                "date": data.get("date", ""),

                "source": "Frankfurter (ECB reference rates)",

            }

    except Exception as exc:

        print("Frankfurter error:", repr(exc))

    if result is None:

        try:

            data = await _http_json(f"{ER_API_URL}/{quote(base)}")

            rate = ((data or {}).get("rates") or {}).get(target)

            if rate:

                result = {

                    "rate": float(rate),

                    "date": str(data.get("time_last_update_utc", ""))[:16],

                    "source": "open.er-api.com",

                }

        except Exception as exc:

            print("ER-API error:", repr(exc))

    if result:

        _cache_set(key, result, 6 * 3600)

    return result

# ============================================================

# Public holidays

# ============================================================

async def public_holidays(

    country_code: str,

    start: date,

    end: date,

) -> list[dict] | None:

    """Public holidays between start and end (Nager.Date)."""

    if not country_code or not start or not end:

        return None

    out: list[dict] = []

    for year in sorted({start.year, end.year}):

        key = ("holidays", country_code.upper(), year)

        hit, items = _cache_get(key)

        if not hit:

            items = await _http_json(f"{NAGER_URL}/{year}/{country_code.upper()}")

            _cache_set(key, items or [], 24 * 3600)

        for h in items or []:

            try:

                h_date = date.fromisoformat(h["date"])

            except (KeyError, ValueError):

                continue

            if start <= h_date <= end:

                out.append(

                    {

                        "date": h["date"],

                        "name": h.get("name", ""),

                        "local_name": h.get("localName", ""),

                    }

                )

    return sorted(out, key=lambda x: x["date"])

# ============================================================

# Weather

# ============================================================

async def weather_forecast(

    lat: float,

    lon: float,

    start: date,

    end: date,

) -> list[dict]:

    """Daily forecast (real forecast, valid up to ~14 days ahead)."""

    params = {

        "latitude": lat,

        "longitude": lon,

        "daily": (

            "weather_code,temperature_2m_max,temperature_2m_min,"

            "precipitation_sum,precipitation_probability_max,"

            "wind_speed_10m_max"

        ),

        "timezone": "auto",

        "start_date": start.isoformat(),

        "end_date": end.isoformat(),

    }

    data = await _http_json(FORECAST_URL, params)

    daily = (data or {}).get("daily") or {}

    days = daily.get("time") or []

    out = []

    for i, d in enumerate(days):

        def pick(key):

            values = daily.get(key) or []

            return values[i] if i < len(values) else None

        out.append(

            {

                "date": d,

                "code": pick("weather_code"),

                "tmax": pick("temperature_2m_max"),

                "tmin": pick("temperature_2m_min"),

                "precip_mm": pick("precipitation_sum"),

                "precip_prob": pick("precipitation_probability_max"),

                "wind_max": pick("wind_speed_10m_max"),

            }

        )

    return out

async def weather_climate(

    lat: float,

    lon: float,

    start: date,

    end: date,

    years: int = 3,

) -> dict | None:

    """

    Climate normals for the SAME calendar dates in previous years

    (Open-Meteo historical archive). Used when the trip is beyond the

    forecast window.

    """

    async def one(back: int):

        params = {

            "latitude": lat,

            "longitude": lon,

            "start_date": _shift_year(start, -back).isoformat(),

            "end_date": _shift_year(end, -back).isoformat(),

            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",

            "timezone": "auto",

        }

        data = await _http_json(ARCHIVE_URL, params, timeout=30.0)

        return (data or {}).get("daily") or {}

    results = await asyncio.gather(

        *[one(y) for y in range(1, years + 1)],

        return_exceptions=True,

    )

    tmax: list[float] = []

    tmin: list[float] = []

    precip: list[float] = []

    used = 0

    for res in results:

        if isinstance(res, Exception):

            print("Climate archive error:", repr(res))

            continue

        used += 1

        tmax += [v for v in res.get("temperature_2m_max", []) if v is not None]

        tmin += [v for v in res.get("temperature_2m_min", []) if v is not None]

        precip += [v for v in res.get("precipitation_sum", []) if v is not None]

    if not tmax or not tmin:

        return None

    return {

        "avg_max": sum(tmax) / len(tmax),

        "avg_min": sum(tmin) / len(tmin),

        "avg_precip_mm": (sum(precip) / len(precip)) if precip else 0.0,

        "rainy_day_ratio": (

            sum(1 for p in precip if p >= 1.0) / len(precip) if precip else 0.0

        ),

        "years_used": used,

        "start": start.isoformat(),

        "end": end.isoformat(),

    }

# ============================================================

# OpenStreetMap data

# ============================================================

async def _overpass(query: str) -> list[dict]:

    last_exc: Exception | None = None

    for endpoint in OVERPASS_ENDPOINTS:

        try:

            data = await _http_json(

                endpoint,

                method="POST",

                data={"data": query},

                timeout=40.0,

            )

            return (data or {}).get("elements", [])

        except Exception as exc:

            last_exc = exc

            print("Overpass error:", repr(exc))

    raise last_exc or RuntimeError("Overpass unavailable")

def _element_latlon(el: dict) -> tuple[float | None, float | None]:

    if "lat" in el and "lon" in el:

        return el["lat"], el["lon"]

    center = el.get("center") or {}

    return center.get("lat"), center.get("lon")

async def osm_hotels(

    lat: float,

    lon: float,

    radius_m: int = 5000,

    limit: int = 12,

) -> list[dict]:

    """Real, named hotels from OpenStreetMap (no prices)."""

    key = ("osm_hotels", round(lat, 2), round(lon, 2), radius_m)

    hit, value = _cache_get(key)

    if hit:

        return value[:limit]

    query = f"""

[out:json][timeout:30];

nwr["tourism"~"^(hotel|guest_house|hostel)$"]["name"](around:{radius_m},{lat},{lon});

out center tags 120;

"""

    elements = await _overpass(query)

    items: list[dict] = []

    seen: set[str] = set()

    for el in elements:

        tags = el.get("tags") or {}

        name = tags.get("name")

        if not name or name.lower() in seen:

            continue

        seen.add(name.lower())

        e_lat, e_lon = _element_latlon(el)

        if e_lat is None or e_lon is None:

            continue

        stars = tags.get("stars", "")

        score = 0

        if stars:

            score += 3

        if tags.get("website") or tags.get("contact:website"):

            score += 2

        if tags.get("tourism") == "hotel":

            score += 1

        items.append(

            {

                "name": name,

                "type": tags.get("tourism", "hotel"),

                "stars": stars,

                "website": tags.get("website") or tags.get("contact:website", ""),

                "street": tags.get("addr:street", ""),

                "lat": e_lat,

                "lon": e_lon,

                "dist_km": round(haversine_km(lat, lon, e_lat, e_lon), 1),

                "score": score,

            }

        )

    items.sort(key=lambda x: (-x["score"], x["dist_km"]))

    _cache_set(key, items, 24 * 3600)

    return items[:limit]

async def osm_attractions(

    lat: float,

    lon: float,

    radius_m: int = 9000,

    limit: int = 24,

) -> list[dict]:

    """

    Notable attractions: OSM objects that have a name AND a wikidata /

    wikipedia tag (a good proxy for 'actually notable').

    """

    key = ("osm_attr", round(lat, 2), round(lon, 2), radius_m)

    hit, value = _cache_get(key)

    if hit:

        return value[:limit]

    query = f"""

[out:json][timeout:35];

(

  nwr["tourism"~"^(attraction|museum|gallery|viewpoint|zoo|theme_park|aquarium)$"]["name"]["wikidata"](around:{radius_m},{lat},{lon});

  nwr["historic"~"^(castle|monument|memorial|ruins|archaeological_site|fort|palace)$"]["name"]["wikidata"](around:{radius_m},{lat},{lon});

);

out center tags 150;

"""

    elements = await _overpass(query)

    items: list[dict] = []

    seen: set[str] = set()

    for el in elements:

        tags = el.get("tags") or {}

        name = tags.get("name:en") or tags.get("name")

        if not name or name.lower() in seen:

            continue

        seen.add(name.lower())

        e_lat, e_lon = _element_latlon(el)

        if e_lat is None or e_lon is None:

            continue

        kind = tags.get("tourism") or tags.get("historic") or "attraction"

        score = 0

        if tags.get("wikipedia"):

            score += 2

        if tags.get("wikidata"):

            score += 1

        if tags.get("website"):

            score += 1

        if kind in ("museum", "castle", "palace", "attraction"):

            score += 1

        items.append(

            {

                "name": name,

                "kind": kind,

                "lat": e_lat,

                "lon": e_lon,

                "dist_km": round(haversine_km(lat, lon, e_lat, e_lon), 1),

                "score": score,

            }

        )

    items.sort(key=lambda x: (-x["score"], x["dist_km"]))

    _cache_set(key, items, 24 * 3600)

    return items[:limit]

# ============================================================

# Airport data

# ============================================================

_AIRPORT_FILE = Path(__file__).resolve().parent / ".cache" / "airports.csv"

_airports_mem: list[dict] | None = None

def _parse_airports(text: str) -> list[dict]:

    rows: list[dict] = []

    for row in csv.DictReader(io.StringIO(text)):

        iata = (row.get("iata_code") or "").strip()

        if not iata:

            continue

        if row.get("type") not in ("large_airport", "medium_airport"):

            continue

        if row.get("scheduled_service") != "yes":

            continue

        try:

            a_lat = float(row["latitude_deg"])

            a_lon = float(row["longitude_deg"])

        except (KeyError, ValueError, TypeError):

            continue

        rows.append(

            {

                "iata": iata,

                "name": row.get("name", ""),

                "city": row.get("municipality", ""),

                "country": row.get("iso_country", ""),

                "type": row.get("type", ""),

                "lat": a_lat,

                "lon": a_lon,

            }

        )

    return rows

async def _load_airports() -> list[dict]:

    global _airports_mem

    if _airports_mem is not None:

        return _airports_mem

    fresh = (

        _AIRPORT_FILE.exists()

        and (time.time() - _AIRPORT_FILE.stat().st_mtime) < 30 * 86400

    )

    if not fresh:

        try:

            async with httpx.AsyncClient(

                timeout=90.0,

                headers=HTTP_HEADERS,

                follow_redirects=True,

            ) as http:

                resp = await http.get(AIRPORTS_URL)

                resp.raise_for_status()

            _AIRPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

            _AIRPORT_FILE.write_text(resp.text, encoding="utf-8")

        except Exception as exc:

            print("Airport dataset download failed:", repr(exc))

            if not _AIRPORT_FILE.exists():

                return []

    text = _AIRPORT_FILE.read_text(encoding="utf-8", errors="ignore")

    _airports_mem = await asyncio.to_thread(_parse_airports, text)

    return _airports_mem

async def nearest_airports(

    lat: float,

    lon: float,

    max_km: float = 200.0,

    limit: int = 4,

) -> list[dict]:

    """

    Scheduled-service airports near a point. Large international airports

    within range are preferred over a closer small airport.

    """

    airports = await _load_airports()

    near: list[dict] = []

    for a in airports:

        dist = haversine_km(lat, lon, a["lat"], a["lon"])

        if dist <= max_km:

            near.append({**a, "dist_km": round(dist, 1)})

    near.sort(

        key=lambda a: (0 if a["type"] == "large_airport" else 1, a["dist_km"])

    )

    return near[:limit]