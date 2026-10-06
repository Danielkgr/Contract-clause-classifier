"""
List prices, and the cost of one call worked out from the usage it reports.

Every price is in US dollars per million tokens.  A model missing from
PRICES is reported as not priced, rather than charged at a guessed rate.
"""

from dataclasses import dataclass

# The Message Batches API bills every token at half the listed rate
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class Price:
    """Per-million-token prices for one model."""

    input: float
    output: float  # thinking tokens are billed as output
    cache_read: float | None = None  # None means no separate rate is listed
    cache_write: float | None = None  # writes to the default 5-minute cache


PRICES = {
    # Anthropic first-party API list prices, checked on 2026-10-06.  Cache
    # writes to the 5-minute cache cost 1.25 times the input price.
    "claude-opus-5-5": Price(4.00, 20.00, cache_read=0.20, cache_write=5.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, cache_read=0.20, cache_write=2.50),
    "claude-haiku-4-5": Price(1.00, 5.00, cache_read=0.10, cache_write=1.25),
    "claude-fable-5-1": Price(10.00, 50.00, cache_read=0.25, cache_write=12.50),
    # OpenAI list prices as recorded in this repository on 2026-09-23 and not
    # checked since.  Check https://openai.com/api/pricing before relying on
    # them.  Any discount OpenAI applies to cached input is not modelled.
    "gpt-3.5-turbo": Price(0.50, 1.50),
    "gpt-3.5-turbo-16k": Price(3.00, 4.00),
    "gpt-4": Price(30.00, 60.00),
    "gpt-4-turbo": Price(10.00, 30.00),
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4o-mini": Price(0.15, 0.60),
}


def price_for(model: str) -> Price | None:
    """The listed price for a model, or None when it has none."""
    return PRICES.get(model)


def cost_usd(
    price: Price | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> float | None:
    """Cost of one call from its token usage, or None when the model is not priced.

    input_tokens is the input billed at the full rate.  Cached input is billed
    at its own rate where one is listed, and otherwise at the input rate.
    """
    if price is None:
        return None
    read_rate = price.input if price.cache_read is None else price.cache_read
    write_rate = price.input if price.cache_write is None else price.cache_write
    total = (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_tokens * read_rate
        + cache_write_tokens * write_rate
    ) / 1_000_000
    return total * BATCH_DISCOUNT if batch else total
