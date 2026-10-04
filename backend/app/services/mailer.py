"""Outgoing email, with a usable path when no mail server is configured.

Registration is pointless if the activation link cannot be delivered, and a
local install usually has no SMTP server. So when one is not configured the
link is logged, and outside production it is also handed back to the caller —
visible in the API response and in the UI — which keeps the whole flow
testable on a laptop. That shortcut is closed in production, where returning an
activation link to whoever asked would let anyone activate any address.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Delivery:
    sent: bool
    detail: str
    # Only populated when there is no mail server and this is not production.
    link: str | None = None


def send_activation_email(to_address: str, name: str, link: str) -> Delivery:
    subject = "Activate your ResearchCompass account"
    minutes = settings.activation_token_minutes

    text = (
        f"Hello {name},\n\n"
        "Confirm this address to finish setting up your ResearchCompass account:\n\n"
        f"{link}\n\n"
        f"The link expires in {minutes} minutes. If it does, sign in and request a new one.\n\n"
        "If you did not create this account, ignore this message — nothing happens\n"
        "until the link is used.\n"
    )

    html = f"""\
<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a">
  <h2 style="margin:0 0 12px">Confirm your email</h2>
  <p>Hello {name}, confirm this address to finish setting up your ResearchCompass account.</p>
  <p style="margin:24px 0">
    <a href="{link}" style="background:#2563eb;color:#fff;padding:11px 20px;
       border-radius:6px;text-decoration:none;display:inline-block">Activate my account</a>
  </p>
  <p style="color:#64748b;font-size:14px">
    The link expires in {minutes} minutes. If it does, sign in and request a new one.
  </p>
  <p style="color:#64748b;font-size:13px;word-break:break-all">{link}</p>
  <p style="color:#94a3b8;font-size:12px">
    If you did not create this account, ignore this message — nothing happens until
    the link is used.
  </p>
</body></html>"""

    return _send(to_address, subject, text, html, link)


def send_password_changed_email(to_address: str, name: str) -> Delivery:
    """Told after the fact, because a password change the owner did not make is
    the one thing they need to hear about immediately."""
    subject = "Your ResearchCompass password was changed"
    text = (
        f"Hello {name},\n\n"
        "The password on your ResearchCompass account was just changed.\n\n"
        "If this was not you, reset it immediately — whoever made the change can "
        "sign in until you do.\n"
    )
    return _send(to_address, subject, text, None, None)


def _send(
    to_address: str, subject: str, text: str, html: str | None, link: str | None
) -> Delivery:
    if not settings.smtp_configured:
        logger.warning(
            "No SMTP server configured; %r was not emailed to %s. Link: %s",
            subject,
            to_address,
            link or "(none)",
        )
        expose = settings.environment != "production"
        return Delivery(
            sent=False,
            detail=(
                "No mail server is configured, so the email was not sent. "
                + (
                    "Use the link below to continue."
                    if expose and link
                    else "Set SMTP_HOST in backend/.env."
                )
            ),
            link=link if expose else None,
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from
    message["To"] = to_address
    message.set_content(text)
    if html:
        message.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as server:
            if settings.smtp_use_tls:
                server.starttls()
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
    except Exception as exc:
        # A failed send must not lose the account that was just created: the
        # user can ask for another link, and in development the link is still
        # reachable from the log.
        logger.exception("Could not send %r to %s", subject, to_address)
        expose = settings.environment != "production"
        return Delivery(
            sent=False,
            detail=f"The email could not be sent ({type(exc).__name__}). Request another link.",
            link=link if expose else None,
        )

    logger.info("Sent %r to %s", subject, to_address)
    return Delivery(sent=True, detail=f"Sent to {to_address}.")
