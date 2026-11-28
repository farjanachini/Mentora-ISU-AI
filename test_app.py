import json
import io
import os
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import patch

import app


def _authenticated_client():
    client = app.app.test_client()
    with client.session_transaction() as app_session:
        app_session["user_id"] = 1
        app_session["user_name"] = "Test User"
    return client


def _csrf_token(response):
    match = re.search(
        r'name="csrf_token" value="([^"]+)"',
        response.get_data(as_text=True),
    )
    if not match:
        raise AssertionError("The page did not include a CSRF token.")
    return match.group(1)


class ChatApiTests(unittest.TestCase):
    def test_chat_includes_previous_turns_in_model_request(self):
        captured = {}
        history = [
            {"role": "user", "content": "Hola"},
            {"role": "model", "content": "¡Hola! ¿En qué puedo ayudarte?"},
        ]

        def generate_content(**kwargs):
            captured["contents"] = kwargs["contents"]
            captured["config"] = kwargs["config"]
            return SimpleNamespace(text="It is a four-year program.")

        with patch.object(
            app, "client",
            SimpleNamespace(models=SimpleNamespace(
                generate_content=generate_content
            ))
        ), patch.object(app, "retrieve_documents", return_value=""):
            response = _authenticated_client().post(
                "/api/chat",
                data={
                    "message": "入学手続きについて教えてください。",
                    "history": json.dumps(history),
                }
            )

        self.assertEqual(response.status_code, 200)
        contents = captured["contents"]
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "Hola")
        self.assertEqual(contents[1].role, "model")
        self.assertEqual(
            contents[1].parts[0].text,
            "¡Hola! ¿En qué puedo ayudarte?"
        )
        self.assertIn(
            "User Question:\n入学手続きについて教えてください。",
            contents[2]
        )
        self.assertIn(
            "required response language for this turn is Japanese",
            contents[2]
        )
        self.assertIn(
            "required response language for this turn is Japanese",
            captured["config"].system_instruction
        )

    def test_detects_short_spanish_greeting(self):
        self.assertEqual(app._detect_language("Hola"), "Spanish")

    def test_defaults_to_english(self):
        self.assertEqual(app._response_language("", []), "English")

    def test_explicit_language_request_overrides_question_language(self):
        self.assertEqual(
            app._response_language("Please speak in Bangla: what is admission?", []),
            "Bengali"
        )

    def test_explicit_language_request_supports_direct_language_form(self):
        self.assertEqual(
            app._response_language("Answer me in French, please.", []),
            "French"
        )

    def test_detects_user_language_before_history(self):
        history = [{"role": "user", "content": "Hola"}]
        self.assertEqual(
            app._response_language("こんにちは", history),
            "Japanese"
        )

    def test_uses_current_question_language_not_history(self):
        history = [{"role": "user", "content": "Hola"}]
        self.assertEqual(
            app._response_language("ভর্তি কীভাবে হবে?", history),
            "Bengali"
        )

    def test_common_english_question_is_not_detected_as_swahili(self):
        self.assertEqual(
            app._detect_language("What is the admission process?"),
            "English"
        )

    def test_all_faculty_question_uses_room_for_complete_list(self):
        captured_config = {}

        def generate_content(**kwargs):
            captured_config["config"] = kwargs["config"]
            return SimpleNamespace(
                text="Prof. A — Professor.\nDr. B — Lecturer."
            )

        with patch.object(
            app, "client",
            SimpleNamespace(models=SimpleNamespace(
                generate_content=generate_content
            ))
        ), patch.object(
            app, "retrieve_documents",
            return_value="Source URL: https://isu.ac.bd/faculties/cse"
        ):
            response = _authenticated_client().post(
                "/api/chat",
                data={"message": "List all CSE faculty members"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Dr. B", response.get_json()["response"])
        self.assertGreaterEqual(
            captured_config["config"].max_output_tokens,
            1000
        )

    def test_greeting_skips_website_retrieval_and_has_no_sources(self):
        fake_models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text="Hello! How can I help?"
            )
        )

        with patch.object(
            app, "client", SimpleNamespace(models=fake_models)
        ), patch.object(app, "retrieve_documents") as retrieve_documents:
            response = _authenticated_client().post(
                "/api/chat",
                data={"message": "Hi!"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["sources"], [])
        self.assertNotIn("grounding", response.get_json())
        retrieve_documents.assert_not_called()

    def test_chat_adds_matched_adjunct_category_if_model_omits_it(self):
        context = "\n".join([
            "Source URL: https://isu.ac.bd/faculties/department-of-computer-science-and-engineering",
            "Information:",
            "Page heading: Faculty Members (Department of Computer Science & Engineering)",
            "Faculty category: Adjunct Faculties",
            "Name: Md. Wahiduzzaman Wahid",
            "Personal title: Chief Technology Officer, Standard Group",
        ])
        fake_models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text=(
                    "Role: Chief Technology Officer, Standard Group. "
                    "Qualifications: BUET, DU, OCP."
                )
            )
        )

        with patch.object(
            app, "client", SimpleNamespace(models=fake_models)
        ), patch.object(
            app, "retrieve_documents", return_value=context
        ):
            response = _authenticated_client().post(
                "/api/chat",
                data={"message": "Tell me about Wahiduzzaman Wahid"}
            )

        self.assertEqual(response.status_code, 200)
        answer = response.get_json()["response"]
        self.assertIn("an adjunct faculty member", answer)
        self.assertIn("Department of Computer Science & Engineering", answer)


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.previous_database = app.app.config["DATABASE"]
        self.temp_directory = tempfile.TemporaryDirectory()
        app.app.config["DATABASE"] = os.path.join(
            self.temp_directory.name,
            "test-users.db",
        )
        app.initialize_database()
        self.client = app.app.test_client()

    def tearDown(self):
        app.app.config["DATABASE"] = self.previous_database
        self.temp_directory.cleanup()

    def test_chat_requires_login(self):
        response = self.client.post("/api/chat", data={"message": "Hello"})
        self.assertEqual(response.status_code, 401)

    def test_guest_login_allows_chat_without_creating_account(self):
        form = self.client.get("/login")
        response = self.client.post(
            "/guest",
            data={
                "csrf_token": _csrf_token(form),
                "next": "/",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/")
        self.assertEqual(self.client.get("/").status_code, 200)
        with patch.object(app, "client", None):
            chat_response = self.client.post(
                "/api/chat",
                data={"message": "Hello"},
            )
        self.assertEqual(chat_response.status_code, 503)
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            user_count = connection.execute(
                "SELECT COUNT(*) FROM users"
            ).fetchone()[0]
        self.assertEqual(user_count, 0)

    def test_registration_creates_hashed_account_and_logs_in(self):
        form = self.client.get("/register")
        response = self.client.post(
            "/register",
            data={
                "csrf_token": _csrf_token(form),
                "name": "Ada Lovelace",
                "email": "Ada@example.com",
                "password": "a-strong-password",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/")
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            row = connection.execute(
                "SELECT name, email, password_hash FROM users"
            ).fetchone()
        self.assertEqual(row[:2], ("Ada Lovelace", "ada@example.com"))
        self.assertNotEqual(row[2], "a-strong-password")
        self.assertTrue(
            app.check_password_hash(row[2], "a-strong-password")
        )
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_registration_rejects_duplicate_email_case_insensitively(self):
        first_form = self.client.get("/register")
        self.client.post(
            "/register",
            data={
                "csrf_token": _csrf_token(first_form),
                "name": "Ada Lovelace",
                "email": "ada@example.com",
                "password": "a-strong-password",
            },
        )

        second_client = app.app.test_client()
        second_form = second_client.get("/register")
        response = second_client.post(
            "/register",
            data={
                "csrf_token": _csrf_token(second_form),
                "name": "Another Ada",
                "email": "ADA@example.com",
                "password": "another-strong-password",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "An account with that email already exists.",
            response.get_data(as_text=True),
        )

    def test_login_rejects_wrong_password_and_accepts_correct_password(self):
        form = self.client.get("/register")
        self.client.post(
            "/register",
            data={
                "csrf_token": _csrf_token(form),
                "name": "Ada Lovelace",
                "email": "ada@example.com",
                "password": "a-strong-password",
            },
        )
        home_page = self.client.get("/")
        logout_response = self.client.post(
            "/logout",
            data={"csrf_token": _csrf_token(home_page)},
        )
        self.assertEqual(logout_response.status_code, 302)

        login_form = self.client.get("/login")
        invalid_response = self.client.post(
            "/login",
            data={
                "csrf_token": _csrf_token(login_form),
                "email": "ada@example.com",
                "password": "incorrect-password",
            },
        )
        self.assertEqual(invalid_response.status_code, 200)

        login_form = self.client.get("/login")
        valid_response = self.client.post(
            "/login",
            data={
                "csrf_token": _csrf_token(login_form),
                "email": "ADA@example.com",
                "password": "a-strong-password",
            },
        )
        self.assertEqual(valid_response.status_code, 302)
        self.assertEqual(valid_response.location, "/")
        self.assertEqual(self.client.get("/").status_code, 200)


class AdminRoleTests(unittest.TestCase):
    def setUp(self):
        self.previous_database = app.app.config["DATABASE"]
        self.previous_knowledge_folder = app.app.config["KNOWLEDGE_FOLDER"]
        self.temp_directory = tempfile.TemporaryDirectory()
        app.app.config["DATABASE"] = os.path.join(
            self.temp_directory.name,
            "test-users.db",
        )
        app.app.config["KNOWLEDGE_FOLDER"] = os.path.join(
            self.temp_directory.name,
            "knowledge",
        )
        os.makedirs(app.app.config["KNOWLEDGE_FOLDER"], exist_ok=True)
        app.initialize_database()

    def tearDown(self):
        app.app.config["DATABASE"] = self.previous_database
        app.app.config["KNOWLEDGE_FOLDER"] = self.previous_knowledge_folder
        self.temp_directory.cleanup()

    def _create_user(self, name, email, role="user"):
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (name, email, password_hash, role)
                VALUES (?, ?, ?, ?)
                """,
                (
                    name,
                    email,
                    app.generate_password_hash("test-password"),
                    role,
                ),
            )
            connection.commit()
            return cursor.lastrowid

    def _client_for_user(self, user_id):
        client = app.app.test_client()
        with client.session_transaction() as app_session:
            app_session["user_id"] = user_id
            app_session["user_name"] = "Test User"
        return client

    def test_database_migration_preserves_existing_users_as_regular_users(self):
        database_path = app.app.config["DATABASE"]
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute("DROP TABLE users")
            connection.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                ("Existing User", "existing@example.com", "hashed-password"),
            )
            connection.commit()

        app.initialize_database()

        with closing(sqlite3.connect(database_path)) as connection:
            user = connection.execute(
                "SELECT name, role FROM users"
            ).fetchone()
        self.assertEqual(user, ("Existing User", "user"))

    def test_database_migration_normalizes_and_deduplicates_legacy_emails(self):
        database_path = app.app.config["DATABASE"]
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute("DROP TABLE users")
            connection.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                [
                    ("First Account", " Ada@Example.com ", "hashed-one"),
                    ("Duplicate Account", "ada@example.com", "hashed-two"),
                ],
            )
            connection.commit()

        app.initialize_database()

        with closing(sqlite3.connect(database_path)) as connection:
            rows = connection.execute(
                "SELECT name, email FROM users"
            ).fetchall()
        self.assertEqual(
            rows,
            [("First Account", "ada@example.com")],
        )

    def test_database_rejects_duplicate_email_across_roles(self):
        first_id = self._create_user("First User", "shared@example.com")
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO users (name, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "Duplicate Admin",
                        "SHARED@example.com",
                        app.generate_password_hash("test-password"),
                        "admin",
                    ),
                )
        self.assertEqual(app._get_user_role(first_id), "user")

    def test_admin_can_upload_and_delete_knowledge_file(self):
        admin_id = self._create_user(
            "Knowledge Admin",
            "knowledge-admin@example.com",
            role="admin",
        )
        client = self._client_for_user(admin_id)
        form = client.get("/admin/knowledge")
        response = client.post(
            "/admin/knowledge/upload",
            data={
                "csrf_token": _csrf_token(form),
                "file": (
                    io.BytesIO(b"Admission starts in January."),
                    "admission.txt",
                ),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 302)
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            row = connection.execute(
                "SELECT id, original_name FROM knowledge_files"
            ).fetchone()
        self.assertEqual(row[1], "admission.txt")

        form = client.get("/admin/knowledge")
        delete_response = client.post(
            f"/admin/knowledge/{row[0]}/delete",
            data={"csrf_token": _csrf_token(form)},
        )
        self.assertEqual(delete_response.status_code, 302)
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM knowledge_files"
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_regular_user_cannot_open_knowledge_library(self):
        regular_id = self._create_user(
            "Regular User",
            "regular@example.com",
        )
        client = self._client_for_user(regular_id)

        response = client.get("/admin/knowledge")

        self.assertEqual(response.status_code, 403)

    def test_knowledge_file_is_added_to_regular_user_chat_context(self):
        admin_id = self._create_user(
            "Knowledge Admin",
            "knowledge-admin@example.com",
            role="admin",
        )
        regular_id = self._create_user(
            "Regular User",
            "regular@example.com",
        )
        admin_client = self._client_for_user(admin_id)
        form = admin_client.get("/admin/knowledge")
        upload_response = admin_client.post(
            "/admin/knowledge/upload",
            data={
                "csrf_token": _csrf_token(form),
                "file": (
                    io.BytesIO(b"Scholarship applications close on 15 June."),
                    "scholarship.txt",
                ),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(upload_response.status_code, 302)

        captured = {}

        def generate_content(**kwargs):
            captured["contents"] = kwargs["contents"]
            return SimpleNamespace(
                text="The scholarship applications close on 15 June."
            )

        regular_client = self._client_for_user(regular_id)
        with patch.object(
            app,
            "client",
            SimpleNamespace(models=SimpleNamespace(
                generate_content=generate_content
            )),
        ), patch.object(app, "retrieve_documents", return_value=""):
            chat_response = regular_client.post(
                "/api/chat",
                data={"message": "When does scholarship application end?"},
            )

        self.assertEqual(chat_response.status_code, 200)
        context = "\n".join(
            str(part)
            for part in captured["contents"]
        )
        self.assertIn(
            "Scholarship applications close on 15 June.",
            context,
        )

    def test_guest_chat_uses_website_context_but_not_admin_knowledge_or_history(self):
        stored_name = "private-knowledge.txt"
        with open(
            os.path.join(app.app.config["KNOWLEDGE_FOLDER"], stored_name),
            "w",
            encoding="utf-8",
        ) as knowledge_file:
            knowledge_file.write("Private internal scholarship code: BLUEBIRD.")
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            connection.execute(
                """
                INSERT INTO knowledge_files (
                    original_name, stored_name, mime_type, size_bytes
                )
                VALUES (?, ?, ?, ?)
                """,
                ("private-knowledge.txt", stored_name, "text/plain", 48),
            )
            connection.commit()

        guest_client = app.app.test_client()
        with guest_client.session_transaction() as guest_session:
            guest_session["is_guest"] = True
            guest_session["user_name"] = "Guest"

        captured = {}

        def generate_content(**kwargs):
            captured["contents"] = kwargs["contents"]
            captured["system_instruction"] = kwargs["config"].system_instruction
            return SimpleNamespace(text="The website lists the available program.")

        with patch.object(
            app,
            "client",
            SimpleNamespace(models=SimpleNamespace(
                generate_content=generate_content
            )),
        ), patch.object(
            app,
            "retrieve_documents",
            return_value=(
                "Source URL: https://isu.ac.bd/programs\n"
                "Information: The website lists the available program."
            ),
        ):
            response = guest_client.post(
                "/api/chat",
                data={
                    "message": "What programs are available?",
                    "history": json.dumps([
                        {"role": "model", "content": "Private prior answer"}
                    ]),
                },
            )

        self.assertEqual(response.status_code, 200)
        context = "\n".join(str(part) for part in captured["contents"])
        self.assertIn("The website lists the available program.", context)
        self.assertNotIn("BLUEBIRD", context)
        self.assertNotIn("Private prior answer", context)
        self.assertIn("Use only facts present in the official website", captured["system_instruction"])

    def test_guest_cannot_upload_chat_files(self):
        guest_client = app.app.test_client()
        with guest_client.session_transaction() as guest_session:
            guest_session["is_guest"] = True

        response = guest_client.post(
            "/api/chat",
            data={
                "message": "Summarize this.",
                "file": (io.BytesIO(b"private information"), "private.txt"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 403)

    def test_reset_users_keeps_only_admin_accounts(self):
        admin_id = self._create_user(
            "Existing Admin",
            "admin@example.com",
            role="admin",
        )
        self._create_user("Regular One", "regular1@example.com")
        self._create_user("Regular Two", "regular2@example.com")
        runner = app.app.test_cli_runner()

        result = runner.invoke(args=["reset-users", "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Deleted 2 non-admin account(s)", result.output)
        with closing(sqlite3.connect(app.app.config["DATABASE"])) as connection:
            rows = connection.execute(
                "SELECT id, email, role FROM users"
            ).fetchall()
        self.assertEqual(
            rows,
            [(admin_id, "admin@example.com", "admin")],
        )

    def test_cli_bootstraps_only_the_first_admin(self):
        user_id = self._create_user("First Admin", "first@example.com")
        runner = app.app.test_cli_runner()

        result = runner.invoke(
            args=["bootstrap-admin", "FIRST@example.com"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(app._get_user_role(user_id), "admin")

        second_result = runner.invoke(
            args=["bootstrap-admin", "first@example.com"],
        )
        self.assertNotEqual(second_result.exit_code, 0)
        self.assertIn("An admin already exists", second_result.output)

    def test_admin_can_promote_and_revoke_with_immediate_session_effect(self):
        first_admin_id = self._create_user(
            "First Admin",
            "first@example.com",
            role="admin",
        )
        second_admin_id = self._create_user(
            "Second Admin",
            "second@example.com",
            role="admin",
        )
        regular_user_id = self._create_user(
            "Regular User",
            "regular@example.com",
        )
        admin_client = self._client_for_user(first_admin_id)
        regular_client = self._client_for_user(regular_user_id)

        page = admin_client.get("/admin/users")
        response = admin_client.post(
            f"/admin/users/{regular_user_id}/role",
            data={
                "csrf_token": _csrf_token(page),
                "action": "promote",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(app._get_user_role(regular_user_id), "admin")
        self.assertEqual(regular_client.get("/admin/users").status_code, 200)

        page = admin_client.get("/admin/users")
        response = admin_client.post(
            f"/admin/users/{second_admin_id}/role",
            data={
                "csrf_token": _csrf_token(page),
                "action": "revoke",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(app._get_user_role(second_admin_id), "user")
        self.assertEqual(
            self._client_for_user(second_admin_id).get("/admin/users").status_code,
            403,
        )

    def test_last_admin_cannot_be_revoked_or_resign(self):
        admin_id = self._create_user(
            "Only Admin",
            "admin@example.com",
            role="admin",
        )
        client = self._client_for_user(admin_id)

        page = client.get("/admin/users")
        revoke_response = client.post(
            f"/admin/users/{admin_id}/role",
            data={
                "csrf_token": _csrf_token(page),
                "action": "revoke",
            },
        )
        self.assertEqual(revoke_response.status_code, 302)
        self.assertEqual(app._get_user_role(admin_id), "admin")

        page = client.get("/admin/users")
        resign_response = client.post(
            "/admin/resign",
            data={"csrf_token": _csrf_token(page)},
        )
        self.assertEqual(resign_response.status_code, 302)
        self.assertEqual(app._get_user_role(admin_id), "admin")
        self.assertEqual(client.get("/admin/users").status_code, 200)

    def test_admin_can_resign_when_another_admin_remains(self):
        resigning_admin_id = self._create_user(
            "Resigning Admin",
            "resigning@example.com",
            role="admin",
        )
        remaining_admin_id = self._create_user(
            "Remaining Admin",
            "remaining@example.com",
            role="admin",
        )
        client = self._client_for_user(resigning_admin_id)
        page = client.get("/admin/users")

        response = client.post(
            "/admin/resign",
            data={"csrf_token": _csrf_token(page)},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/")
        self.assertEqual(app._get_user_role(resigning_admin_id), "user")
        self.assertEqual(app._get_user_role(remaining_admin_id), "admin")
        self.assertEqual(client.get("/admin/users").status_code, 403)

    def test_regular_users_cannot_manage_admin_roles(self):
        regular_user_id = self._create_user(
            "Regular User",
            "regular@example.com",
        )
        response = self._client_for_user(regular_user_id).get("/admin/users")
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
