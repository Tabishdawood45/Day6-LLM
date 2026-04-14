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
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# ─── State ────────────────────────────────────────────────────────────────────

class ProcurementState(TypedDict):
    request: str
    quantity: int
    vendors: list[dict]
    quotes: list[dict]
    best_quote: dict
    approval_status: str
    rejection_reason: str
    po_number: str
    notification: str


# ─── LLM (used only for the notification step to make it feel "agentic") ─────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyAKQtjmfNqa_abBy0DLQsm2sCcthqvsOpE"
)


# ─── Helper functions ─────────────────────────────────────────────────────────

def extract_quantity(request_text: str) -> int:
    """Extract quantity from request string like 'Order 30 laptops for the sales team'."""
    match = re.search(r"(\d+)", request_text)
    return int(match.group(1)) if match else 1


def fetch_laptop_products() -> list[dict]:
    """Fetch laptop products from DummyJSON."""
    url = "https://dummyjson.com/products/category/laptops"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("products", [])
    except Exception as e:
        print(f"   WARNING: Failed to fetch live laptop data from DummyJSON: {e}")
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


# ─── Tool for pricing ─────────────────────────────────────────────────────────

@tool
def get_unit_price(vendor: str) -> float:
    """Get the unit price for a laptop from a vendor."""
    products = fetch_laptop_products()
    best_product = choose_best_live_product(products)

    if best_product:
        return float(best_product["price"])

    # Fallback if no live match is found
    fallback_prices = {
        "Dell": 899.0,
        "Lenovo": 879.0,
        "HP": 929.0,
    }

    print(f"   WARNING: No live product match found. Using fallback price for {vendor}.")
    return fallback_prices.get(vendor, 899.0)


# ─── Node functions ──────────────────────────────────────────────────────────

def lookup_vendors(state: ProcurementState) -> dict:
    """Step 1: Look up approved vendors for laptops."""
    print("\n[Step 1] Looking up approved vendors...")
    time.sleep(1)  # simulate API call
    vendors = [
        {"name": "Dell", "id": "V-001", "category": "laptops", "rating": 4.5},
        {"name": "Lenovo", "id": "V-002", "category": "laptops", "rating": 4.3},
        {"name": "HP", "id": "V-003", "category": "laptops", "rating": 4.1},
    ]
    for v in vendors:
        print(f"   Found vendor: {v['name']} (rating {v['rating']})")
    return {"vendors": vendors}


def fetch_pricing(state: ProcurementState) -> dict:
    """Step 2: Fetch current pricing from all 3 suppliers."""
    print("\n[Step 2] Fetching pricing from suppliers...")
    time.sleep(1.5)  # simulate multiple API calls

    quantity = extract_quantity(state["request"])
    products = fetch_laptop_products()
    best_product = choose_best_live_product(products)

    if best_product:
        product_name = best_product.get("title", "Laptop")
        live_price = float(best_product["price"])
        live_delivery_days = best_product.get("delivery_days", 7)
    else:
        product_name = "Laptop"
        live_delivery_days = 7
        live_price = None
        print("   WARNING: No suitable live product found within 2 weeks. Using fallback prices.")

    # LLM is bound to the tool and asked to call once per vendor
    llm_with_tools = llm.bind_tools([get_unit_price])
    vendor_names = [v["name"] for v in state["vendors"]]

    prompt = (
        f"The employee request is: {state['request']}\n"
        f"Quantity is: {quantity}\n"
        f"Approved vendors are: {', '.join(vendor_names)}\n"
        "You must call the get_unit_price tool exactly once for EACH vendor:\n"
        "- Dell\n"
        "- Lenovo\n"
        "- HP\n"
        "Return tool calls for all three vendors."
    )

    response = llm_with_tools.invoke(prompt)

    quotes = []
    already_called = set()

    delivery_days_map = {
        "Dell": 5,
        "Lenovo": 7,
        "HP": 4,
    }

    # First use tool calls returned by LLM
    if response.tool_calls:
        for tool_call in response.tool_calls:
            vendor = tool_call["args"]["vendor"]

            if vendor in already_called:
                continue

            already_called.add(vendor)

            unit_price = get_unit_price.invoke(tool_call["args"])

            if live_price is not None:
                unit_price = live_price

            total = unit_price * quantity

            quotes.append({
                "vendor": vendor,
                "unit_price": unit_price,
                "total": total,
                "delivery_days": live_delivery_days if live_price is not None else delivery_days_map.get(vendor, 7),
                "product_name": product_name,
            })

    # If LLM missed any vendor, add them manually
    missing_vendors = [vendor for vendor in vendor_names if vendor not in already_called]

    if missing_vendors:
        print(f"   WARNING: LLM missed vendors {missing_vendors}. Adding them manually...")

    for vendor in missing_vendors:
        unit_price = get_unit_price.invoke({"vendor": vendor})

        if live_price is not None:
            unit_price = live_price

        total = unit_price * quantity

        quotes.append({
            "vendor": vendor,
            "unit_price": unit_price,
            "total": total,
            "delivery_days": live_delivery_days if live_price is not None else delivery_days_map.get(vendor, 7),
            "product_name": product_name,
        })

    for q in quotes:
        print(
            f"   {q['vendor']}: €{q['unit_price']}/unit x {quantity} = €{q['total']:,.2f} "
            f"({q['delivery_days']} day delivery) — {q['product_name']}"
        )

    return {
        "quotes": quotes,
        "quantity": quantity,
    }


def compare_quotes(state: ProcurementState) -> dict:
    """Step 3: Compare quotes and pick the best one."""
    print("\n[Step 3] Comparing quotes...")
    time.sleep(0.5)
    best = min(state["quotes"], key=lambda q: q["total"])
    print(f"   Best quote: {best['vendor']} at €{best['total']:,.2f}")
    print(
        f"   (Saves €{max(q['total'] for q in state['quotes']) - best['total']:,.2f} "
        f"vs most expensive option)"
    )
    return {"best_quote": best}


def route_after_compare_quotes(state: ProcurementState) -> Literal["request_approval", "submit_purchase_order"]:
    """Route to approval only if total exceeds €10,000."""
    if state["best_quote"]["total"] > 10_000:
        return "request_approval"
    return "submit_purchase_order"


def request_approval(state: ProcurementState) -> dict:
    """Step 4: Human-in-the-loop — request manager approval for orders > €10,000."""
    best = state["best_quote"]
    quantity = state["quantity"]

    print("\n[Step 4] Order exceeds €10,000 — manager approval required!")
    print(f"   Sending approval request to manager...")
    amount_str = f"€{best['total']:,.2f}"
    delivery_str = f"{best['delivery_days']} business days"
    print(f"   ┌─────────────────────────────────────────────┐")
    print(f"   │  APPROVAL NEEDED                            │")
    print(f"   │  Vendor:   {best['vendor']:<33}│")
    print(f"   │  Product:  {best['product_name'][:33]:<33}│")
    print(f"   │  Amount:   {amount_str:<33}│")
    print(f"   │  Items:    {quantity} x laptops for team    │")
    print(f"   │  Delivery: {delivery_str:<33}│")
    print(f"   └─────────────────────────────────────────────┘")

    # ── THIS IS WHERE THE MAGIC HAPPENS ──
    # interrupt() freezes the entire graph state into the checkpoint store.
    # The process can now exit completely. When resumed later (even days later),
    # execution continues right here with the resume value.
    decision = interrupt({
        "message": (
            f"Approve purchase of {quantity} laptops "
            f"({best['product_name']}) from {best['vendor']} "
            f"for €{best['total']:,.2f}?"
        ),
        "vendor": best["vendor"],
        "amount": best["total"],
        "product_name": best["product_name"],
    })

    print(f"\n[Step 4] Manager responded: {decision}")

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
    """Step 5: Submit the purchase order to the ERP system."""
    print("\n[Step 5] Submitting purchase order to ERP system...")
    time.sleep(1)
    po_number = "PO-2026-00342"
    print(f"   Purchase order created: {po_number}")
    print(f"   Vendor: {state['best_quote']['vendor']}")
    print(f"   Product: {state['best_quote']['product_name']}")
    print(f"   Amount: €{state['best_quote']['total']:,.2f}")
    return {"po_number": po_number}


def notify_employee(state: ProcurementState) -> dict:
    """Step 6: Use LLM to draft and send a notification to the employee."""
    print("\n[Step 6] Notifying employee...")

    quantity = state.get("quantity", 1)
    product_name = state.get("best_quote", {}).get("product_name", "laptops")

    if "reject" in state.get("approval_status", "").lower():
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request for {quantity} {product_name} was rejected by the manager. "
            f"Reason: {state.get('rejection_reason', 'Rejected by manager')}. "
            f"Be empathetic but concise."
        )
    else:
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request has been approved and processed. "
            f"Details: {quantity} {product_name} from {state['best_quote']['vendor']}, "
            f"€{state['best_quote']['total']:,.2f}, PO number {state['po_number']}, "
            f"delivery in {state['best_quote']['delivery_days']} business days."
        )

    response = llm.invoke(prompt)
    notification = response.content
    print(f"   Employee notification sent:")
    print(f"   \"{notification}\"")
    return {"notification": notification}


# ─── Build the graph ─────────────────────────────────────────────────────────
#
#   START → lookup_vendors → fetch_pricing → compare_quotes
#         → request_approval (INTERRUPT)
#         → submit_purchase_order → notify_employee → END

builder = StateGraph(ProcurementState)

builder.add_node("lookup_vendors", lookup_vendors)
builder.add_node("fetch_pricing", fetch_pricing)
builder.add_node("compare_quotes", compare_quotes)
builder.add_node("request_approval", request_approval)
builder.add_node("submit_purchase_order", submit_purchase_order)
builder.add_node("notify_employee", notify_employee)

builder.add_edge(START, "lookup_vendors")
builder.add_edge("lookup_vendors", "fetch_pricing")
builder.add_edge("fetch_pricing", "compare_quotes")

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

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "procurement_checkpoints.db")
THREAD_ID = "procurement-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}


# ─── Main ────────────────────────────────────────────────────────────────────

def run_first_invocation(graph):
    """First run: employee submits request, agent does steps 1-3, then suspends."""
    print("=" * 60)
    print("  FIRST INVOCATION — Employee submits purchase request")
    print("=" * 60)
    print("\nEmployee request: \"Order 30 laptops for the sales team\"")

    result = graph.invoke(
        {"request": "Order 30 laptops for the sales team"},
        config,
    )

    # After interrupt, the graph returns with __interrupt__ info
    print("\n" + "=" * 60)

    if "__interrupt__" in result:
        print("AGENT SUSPENDED — waiting for manager approval")
        print("=" * 60)
        print("\n  The agent process can now exit completely.")
        print("  All state (vendors, pricing, best quote) is frozen in SQLite.")
        print(f"  Checkpoint DB: {DB_PATH}")
        print(f"  Thread ID: {THREAD_ID}")
        print("\n  In a real system, the manager gets a Slack/email notification.")
        print("  They might respond hours or even days later.\n")
        print("  To resume, run:")
        print(f"    python {os.path.basename(__file__)} --resume\n")
    else:
        print("PROCUREMENT COMPLETE (No approval required)")
        print("=" * 60)
        print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
        print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
        print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,.2f}")
        print(f"  Approval:     {result.get('approval_status', 'Not needed')}")
        print()


def run_second_invocation(graph):
    """Second run: manager approves, agent wakes up at step 5 with full context."""
    print("=" * 60)
    print("  SECOND INVOCATION — Manager approves (maybe days later!)")
    print("=" * 60)

    # Show that the state survived the process restart
    saved_state = graph.get_state(config)
    if not saved_state or not saved_state.values:
        print("\nNo saved state found! Run without --resume first.")
        return

    print("\nLoading state from checkpoint...")
    print(f"  ✓ Request: {saved_state.values.get('request', 'N/A')}")
    print(f"  ✓ Quantity: {saved_state.values.get('quantity', 'N/A')}")
    print(f"  ✓ Vendors found: {len(saved_state.values.get('vendors', []))}")
    print(f"  ✓ Quotes received: {len(saved_state.values.get('quotes', []))}")
    best = saved_state.values.get("best_quote", {})
    print(f"  ✓ Best quote: {best.get('vendor', 'N/A')} at €{best.get('total', 0):,.2f}")
    print(f"\n  Steps 1-3 are NOT re-executed — their output is in the checkpoint!\n")

    # Resume with the manager's approval
    # Change the text below to test approval or rejection
    # Example approval:
    #   Command(resume="Approved — go ahead with the purchase.")
    # Example rejection:
    #   Command(resume="Rejected — over budget")
    print("Manager responds ...")
    time.sleep(1)

    result = graph.invoke(
        # Command(resume="Approved — go ahead with the purchase."),
        Command(resume="Rejected — over budget"),
        config,
    )

    print("\n" + "=" * 60)
    print("PROCUREMENT COMPLETE")
    print("=" * 60)
    print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
    print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
    print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,.2f}")
    print(f"  Approval:     {result.get('approval_status', 'N/A')}")
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