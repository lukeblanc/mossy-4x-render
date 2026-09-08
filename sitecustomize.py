"""Legacy recovery hook retired.

Startup must preserve risk baselines and drawdown halts. Application startup
now initializes missing period state explicitly; importing Python never edits
the persisted state or recovery markers.
"""
