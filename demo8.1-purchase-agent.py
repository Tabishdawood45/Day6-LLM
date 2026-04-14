"""
Demo 8 – Resumable AI Procurement Agent (LangGraph Persistence + Interrupt)

Scenario: An AI agent handles purchase requests. When a purchase exceeds
€10,000 it must pause for manager approval — which may come hours or days later.

The graph:

  START → lookup_vendors → fetch_pricing → compare_quotes
        → request_approval (INTERRUPTS here — process exits!)
        → submit_purchase_order → notify_employee → END

To simulate a real-world "late second invocation" across process restarts,
we use SqliteSaver (file-based checkpoint) and two CLI modes:

  python demo8.1-purchase-agent.py              # First run  — steps 1-3, then suspends
  python demo8.1-purchase-agent.py --resume     # Second run — manager approves, steps 5-6

Between the two runs the Python process exits completely.  The full agent
state (vendor data, pricing, chosen quote) survives on disk in SQLite.
"""

import sys
import os
import sqlite3
import time
import re
import requests
import operator
from typing import TypedDict, Literal, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command, Send
from langchain_google_genai import ChatGoogleGenerativeAI

# ─── State ────────────────────────────────────────────────────────────────────

class ProcurementState(TypedDict):
    request: str
    quantity: int
    vendors: list[dict]
    requested_categories: list[str]
    item_quotes: Annotated[list[dict], operator.add]
    best_quote: dict
    approval_status: str
    rejection_reason: str
    po_number: str
    notification: str


class ItemLookupState(TypedDict):
    category: str
    quantity: int


# ─── LLM (used only for the notification step to make it feel "agentic") ─────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)


# ─── Helper functions ─────────────────────────────────────────────────────────

def extract_quantity(request_text: str) -> int:
    """Extract quantity from request string like 'Order 30 laptops for the sales team'."""
    match = re.search(r"(\d+)", request_text)
    return int(match.group(1)) if match else 1


def detect_requested_categories(request_text: str) -> list[str]:
    """
    Detect what items are needed from the request.
    Examples:
      - 'Onboard 10 new engineers — each needs a laptop and a smartphone'
      - 'Onboard 10 new engineers — each needs a laptop'
    """
    text = request_text.lower()
    categories = []

    if "laptop" in text:
        categories.append("laptops")

    if "smartphone" in text or "phone" in text:
        categories.append("smartphones")

    # default for safety if nothing is detected
    if not categories:
        categories.append("laptops")

    return categories


def fetch_products_by_category(category: str) -> list[dict]:
    """Fetch product list from DummyJSON category endpoint."""
    url = f"https://dummyjson.com/products/category/{category}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("products", [])
    except Exception as e:
        print(f"   WARNING: Failed to fetch live {category} data from DummyJSON: {e}")
        return []


def choose_best_live_product(products: list[dict]) -> dict | None:
    """
    Choose the cheapest product that appears available within 2 weeks.
    DummyJSON may not always have delivery_days, so:
      - if delivery_days exists, require <= 14
      - otherwise accept products with stock > 0
    """
    valid = []

    for product in products:
        price = product.get("price")
        stock = product.get("stock", 0)
        delivery_days = product.get("delivery_days", None)

        if price is None:
            continue

        if delivery_days is not None:
            if stock > 0 and delivery_days <= 14:
                valid.append(product)
        else:
            if stock > 0:
                valid.append(product)

    if not valid:
        return None

    return min(valid, key=lambda p: p["price"])


def get_fallback_product(category: str) -> dict:
    """Fallback product if live API fails or no suitable item is found."""
    if category == "smartphones":
        return {
            "title": "Fallback Smartphone",
            "price": 699.0,
            "delivery_days": 7,
            "stock": 100,
        }

    return {
        "title": "Fallback Laptop",
        "price": 899.0,
        "delivery_days": 7,
        "stock": 100,
    }


def category_label(category: str) -> str:
    """Convert API category name to human-readable label."""
    if category == "laptops":
        return "laptop"
    if category == "smartphones":
        return "smartphone"
    return category


# ─── Node functions ──────────────────────────────────────────────────────────

def lookup_vendors(state: ProcurementState) -> dict:
    """Step 1: Look up approved vendors for laptops."""
    print("\n[Step 1] Looking up approved vendors...")
    time.sleep(1)  # simulate API call
    vendors = [
        {"name": "Dell", "id": "V-001", "category": "laptops", "rating": 4.5},
        {"name": "Lenovo", "id": "V-002", "category": "laptops", "rating": 4.3},
        {"name": "HP", "id": "V-003", "category": "laptops", "rating": 4.1},
        {"name": "Samsung", "id": "V-004", "category": "smartphones", "rating": 4.4},
        {"name": "Apple", "id": "V-005", "category": "smartphones", "rating": 4.6},
        {"name": "Xiaomi", "id": "V-006", "category": "smartphones", "rating": 4.2},
    ]
    for v in vendors:
        print(f"   Found vendor: {v['name']} (category {v['category']}, rating {v['rating']})")
    return {"vendors": vendors}


def prepare_item_requests(state: ProcurementState) -> dict:
    """Step 2: Parse onboarding request and decide which item categories are needed."""
    print("\n[Step 2] Parsing onboarding request...")
    time.sleep(0.5)

    quantity = extract_quantity(state["request"])
    requested_categories = detect_requested_categories(state["request"])

    print(f"   Quantity detected: {quantity}")
    print(f"   Categories detected: {', '.join(requested_categories)}")

    return {
        "quantity": quantity,
        "requested_categories": requested_categories,
    }


def route_item_lookups(state: ProcurementState):
    """
    Fan out one pricing lookup per requested category.
    These run in parallel in LangGraph.
    """
    sends = []
    for category in state["requested_categories"]:
        sends.append(
            Send(
                "fetch_item_pricing",
                {
                    "category": category,
                    "quantity": state["quantity"],
                },
            )
        )
    return sends


def fetch_item_pricing(state: ItemLookupState) -> dict:
    """Step 3: Fetch live pricing for one item category."""
    category = state["category"]
    quantity = state["quantity"]

    print(f"\n[Step 3] Fetching pricing for category: {category} ...")
    time.sleep(1)

    products = fetch_products_by_category(category)
    best_product = choose_best_live_product(products)

    if best_product is None:
        print(f"   WARNING: No suitable live {category} product found. Using fallback.")
        best_product = get_fallback_product(category)

    product_name = best_product.get("title", category_label(category).title())
    unit_price = float(best_product.get("price", 0))
    delivery_days = int(best_product.get("delivery_days", 7))
    total = unit_price * quantity

    quote = {
        "category": category,
        "item_label": category_label(category),
        "product_name": product_name,
        "vendor": "Best available market option",
        "unit_price": unit_price,
        "quantity": quantity,
        "delivery_days": delivery_days,
        "total": total,
    }

    print(
        f"   {product_name}: €{unit_price}/unit x {quantity} = "
        f"€{total:,.2f} ({delivery_days} day delivery)"
    )

    # item_quotes uses operator.add reducer in main state
    return {"item_quotes": [quote]}


def compare_quotes(state: ProcurementState) -> dict:
    """Step 4: Compare quotes and build a combined best quote."""
    print("\n[Step 4] Combining item quotes...")
    time.sleep(0.5)

    item_quotes = state.get("item_quotes", [])
    order_total = sum(item["total"] for item in item_quotes)
    max_delivery = max(item["delivery_days"] for item in item_quotes) if item_quotes else 0

    for item in item_quotes:
        print(
            f"   Included: {item['quantity']} x {item['product_name']} "
            f"= €{item['total']:,.2f}"
        )

    best = {
        "vendor": "Multiple suppliers / best market options",
        "items": item_quotes,
        "total": order_total,
        "delivery_days": max_delivery,
    }

    print(f"   Combined order total: €{order_total:,.2f}")
    return {"best_quote": best}


def route_after_compare_quotes(state: ProcurementState) -> Literal["request_approval", "submit_purchase_order"]:
    """Route to approval only if total exceeds €10,000."""
    if state["best_quote"]["total"] > 10_000:
        return "request_approval"
    return "submit_purchase_order"


def request_approval(state: ProcurementState) -> dict:
    """Step 5: Human-in-the-loop — request manager approval for orders > €10,000."""
    best = state["best_quote"]

    print("\n[Step 5] Order exceeds €10,000 — manager approval required!")
    print("   Sending approval request to manager...")

    amount_str = f"€{best['total']:,.2f}"
    delivery_str = f"{best['delivery_days']} business days"

    item_lines = []
    for item in best["items"]:
        item_lines.append(f"{item['quantity']} x {item['product_name']}")

    print(f"   ┌─────────────────────────────────────────────┐")
    print(f"   │  APPROVAL NEEDED                            │")
    print(f"   │  Vendor:   {'Best market options':<33}│")
    print(f"   │  Amount:   {amount_str:<33}│")
    print(f"   │  Delivery: {delivery_str:<33}│")
    print(f"   └─────────────────────────────────────────────┘")
    print("   Items:")
    for line in item_lines:
        print(f"     - {line}")

    decision = interrupt({
        "message": f"Approve onboarding kit purchase for €{best['total']:,.2f}?",
        "amount": best["total"],
        "items": item_lines,
    })

    print(f"\n[Step 5] Manager responded: {decision}")

    if "reject" in str(decision).lower():
        return {
            "approval_status": str(decision),
            "rejection_reason": str(decision),
        }

    return {
        "approval_status": str(decision),
        "rejection_reason": "",
    }


def route_after_approval(state: ProcurementState) -> Literal["submit_purchase_order", "notify_employee"]:
    """If approved continue to PO, if rejected skip PO and notify employee directly."""
    if "reject" in state["approval_status"].lower():
        return "notify_employee"
    return "submit_purchase_order"


def submit_purchase_order(state: ProcurementState) -> dict:
    """Step 6: Submit the purchase order to the ERP system."""
    print("\n[Step 6] Submitting purchase order to ERP system...")
    time.sleep(1)

    po_number = "PO-2026-00999"
    print(f"   Purchase order created: {po_number}")
    print(f"   Total amount: €{state['best_quote']['total']:,.2f}")
    print("   Items in PO:")
    for item in state["best_quote"]["items"]:
        print(
            f"     - {item['quantity']} x {item['product_name']} "
            f"@ €{item['unit_price']}/unit = €{item['total']:,.2f}"
        )

    return {"po_number": po_number}


def notify_employee(state: ProcurementState) -> dict:
    """Step 7: Use LLM to draft and send a notification to the employee."""
    print("\n[Step 7] Notifying employee...")

    items_text = ", ".join(
        f"{item['quantity']} x {item['product_name']}"
        for item in state["best_quote"]["items"]
    )

    if "reject" in state.get("approval_status", "").lower():
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their onboarding purchase request was rejected by the manager. "
            f"Items requested: {items_text}. "
            f"Reason: {state.get('rejection_reason', 'Rejected by manager')}. "
            f"Be empathetic but concise."
        )
    else:
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their onboarding purchase request has been approved and processed. "
            f"Items: {items_text}. "
            f"Total: €{state['best_quote']['total']:,.2f}. "
            f"PO number: {state['po_number']}. "
            f"Delivery in {state['best_quote']['delivery_days']} business days."
        )

    response = llm.invoke(prompt)
    notification = response.content
    print(f"   Employee notification sent:")
    print(f"   \"{notification}\"")
    return {"notification": notification}


# ─── Build the graph ─────────────────────────────────────────────────────────
#
#   START → lookup_vendors → prepare_item_requests
#         → fetch_item_pricing (parallel fanout with Send)
#         → compare_quotes
#         → request_approval (INTERRUPT)
#         → submit_purchase_order → notify_employee → END

builder = StateGraph(ProcurementState)

builder.add_node("lookup_vendors", lookup_vendors)
builder.add_node("prepare_item_requests", prepare_item_requests)
builder.add_node("fetch_item_pricing", fetch_item_pricing)
builder.add_node("compare_quotes", compare_quotes)
builder.add_node("request_approval", request_approval)
builder.add_node("submit_purchase_order", submit_purchase_order)
builder.add_node("notify_employee", notify_employee)

builder.add_edge(START, "lookup_vendors")
builder.add_edge("lookup_vendors", "prepare_item_requests")

builder.add_conditional_edges(
    "prepare_item_requests",
    route_item_lookups,
)

builder.add_edge("fetch_item_pricing", "compare_quotes")

builder.add_conditional_edges(
    "compare_quotes",
    route_after_compare_quotes,
    {
        "request_approval": "request_approval",
        "submit_purchase_order": "submit_purchase_order",
    },
)

builder.add_conditional_edges(
    "request_approval",
    route_after_approval,
    {
        "submit_purchase_order": "submit_purchase_order",
        "notify_employee": "notify_employee",
    },
)

builder.add_edge("submit_purchase_order", "notify_employee")
builder.add_edge("notify_employee", END)


# ─── Checkpointer (SQLite — survives process restarts!) ──────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "procurement_checkpoints_bonus.db")
THREAD_ID = "procurement-bonus-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}


# ─── Main ────────────────────────────────────────────────────────────────────

def run_first_invocation(graph):
    """First run: employee submits request, agent does steps 1-4, then may suspend."""
    print("=" * 60)
    print("  FIRST INVOCATION — Employee submits onboarding request")
    print("=" * 60)
    print('\nEmployee request: "Onboard 10 new engineers — each needs a laptop and a smartphone"')

    result = graph.invoke(
        {"request": "Onboard 10 new engineers — each needs a laptop and a smartphone"},
        config,
    )

    print("\n" + "=" * 60)

    if "__interrupt__" in result:
        print("AGENT SUSPENDED — waiting for manager approval")
        print("=" * 60)
        print("\n  The agent process can now exit completely.")
        print("  All state (item quotes, chosen products, combined total) is frozen in SQLite.")
        print(f"  Checkpoint DB: {DB_PATH}")
        print(f"  Thread ID: {THREAD_ID}")
        print("\n  To resume, run:")
        print(f"    python {os.path.basename(__file__)} --resume\n")
    else:
        print("PROCUREMENT COMPLETE (No approval required)")
        print("=" * 60)
        print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
        print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,.2f}")
        print(f"  Approval:     {result.get('approval_status', 'Not needed')}")
        print()


def run_second_invocation(graph):
    """Second run: manager approves or rejects later."""
    print("=" * 60)
    print("  SECOND INVOCATION — Manager response")
    print("=" * 60)

    saved_state = graph.get_state(config)
    if not saved_state or not saved_state.values:
        print("\nNo saved state found! Run without --resume first.")
        return

    print("\nLoading state from checkpoint...")
    print(f"  ✓ Request: {saved_state.values.get('request', 'N/A')}")
    print(f"  ✓ Quantity: {saved_state.values.get('quantity', 'N/A')}")
    print(f"  ✓ Categories: {saved_state.values.get('requested_categories', [])}")
    print(f"  ✓ Item quotes: {len(saved_state.values.get('item_quotes', []))}")
    print(f"  ✓ Combined total: €{saved_state.values.get('best_quote', {}).get('total', 0):,.2f}")
    print(f"\n  Earlier steps are NOT re-executed — their output is in the checkpoint!\n")

    print("Manager responds ...")
    time.sleep(1)

    result = graph.invoke(
        Command(resume="Approved — go ahead with the purchase."),
        config,
    )

    print("\n" + "=" * 60)
    print("PROCUREMENT COMPLETE")
    print("=" * 60)
    print(f"\n  PO Number:    {result.get('po_number', 'NOT CREATED')}")
    print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,.2f}")
    print(f"  Approval:     {result.get('approval_status', 'N/A')}")
    if result.get("rejection_reason"):
        print(f"  Rejection:    {result.get('rejection_reason')}")
    print()


if __name__ == "__main__":
    resume_mode = "--resume" in sys.argv

    # Clean start if not resuming
    if not resume_mode and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"(Cleaned up old checkpoint DB)")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    try:
        if resume_mode:
            run_second_invocation(graph)
        else:
            run_first_invocation(graph)
    finally:
        conn.close()