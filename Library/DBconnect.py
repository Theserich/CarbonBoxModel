from xml.etree.ElementTree import parse
from base64 import b64decode
from sys import argv

import keyring
from keyring.errors import KeyringError, PasswordDeleteError
from mysql.connector import connect, Error
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi

from Library.GlobalPathsAndConstants import projectPath

# keyring entry that remembers the last used username (the password itself is
# stored under the real username, so several accounts can coexist)
_USERNAME_KEY = '__last_username__'


class DBconnect():
    def __init__(self, myhost='mysql-lip.phys.ethz.ch', port=3306):
        self.__myhost = myhost
        self.__port = port
        self.__service = 'DBconnect:%s:%i' % (myhost, port)
        self.__ok = True
        self.__remember = False
        self.__credentialsChanged = False
        self.__username, self.__password = '', ''
        if not self.__getCredentialsFromKeyring():
            # one-time migration from the old base64 XML file
            self.__migrateLegacyFile(projectPath/'Settings/cred.xml')

    def getConnection(self):
        if self.__ok:
            try:
                cnx = connect(user=self.__username, password=self.__password,
                              host=self.__myhost, port=self.__port)
            except Error as err:
                if err.errno == 1045:
                    self.__showLoginWindow(warning='wrong username or password')
                elif err.errno == 1130:
                    self.__showLoginWindow(warning='no connection to DB')
                else:
                    print(err, err.errno)
                    self.__showLoginWindow(warning='unknown mysql.connector error: %s' % err.errno)
                return self.getConnection()
            else:
                # only store credentials that were actually accepted by the server
                if self.__credentialsChanged:
                    self.__saveCredentialsToKeyring()
                    self.__credentialsChanged = False
                return cnx
        else:
            self.__ok = True
            return None

    def __showLoginWindow(self, warning):
        self.__ok = False
        app = QApplication.instance() or QApplication(argv)  # reuse a running app
        self.__window = loadui(projectPath/'UILogin.ui')
        self.__window.label_warning.setText('<html><head/><body><p align=\"center\"><span style=\" color:#c80000;\">%s</span></p></body></html>' % (warning,))
        self.__window.lineEdit_User.setText(self.__username)
        self.__window.lineEdit_PW.setText(self.__password)
        self.__window.cB_Remember.setChecked(self.__remember)
        self.__window.pushButton_OK.clicked.connect(lambda: self.__pressedOK())
        self.__window.pushButton_Cancel.clicked.connect(lambda: self.__pressedCancel())
        self.__window.lineEdit_User.returnPressed.connect(lambda: self.__pressedOK())
        self.__window.lineEdit_PW.returnPressed.connect(lambda: self.__pressedOK())
        self.__window.exec_()

    def __pressedOK(self):
        self.__username = self.__window.lineEdit_User.text()
        self.__password = self.__window.lineEdit_PW.text()
        self.__remember = self.__window.cB_Remember.isChecked()
        self.__credentialsChanged = True
        self.__ok = True
        self.__window.close()

    def __pressedCancel(self):
        self.__window.close()

    # ---------------------------------------------------------------- keyring
    def __getCredentialsFromKeyring(self):
        try:
            username = keyring.get_password(self.__service, _USERNAME_KEY)
            if username:
                self.__username = username
                password = keyring.get_password(self.__service, username)
                if password is not None:
                    self.__password = password
                    self.__remember = True
                return True
        except KeyringError as err:
            print('keyring not available:', err)
        return False

    def __saveCredentialsToKeyring(self):
        try:
            keyring.set_password(self.__service, _USERNAME_KEY, self.__username)
            if self.__remember:
                keyring.set_password(self.__service, self.__username, self.__password)
            else:
                try:
                    keyring.delete_password(self.__service, self.__username)
                except PasswordDeleteError:
                    pass  # nothing stored
        except KeyringError as err:
            print('could not save credentials to keyring:', err)

    def __migrateLegacyFile(self, file):
        if not file.exists():
            return
        try:
            for login in parse(file).iterfind('login'):
                self.__username = login.find('username').text or ''
                pw = login.find('password')
                if pw is not None and pw.text:
                    self.__password = b64decode(pw.text).decode()
                    self.__remember = True
        except Exception as err:
            print('could not read legacy credentials file:', err)
            return
        self.__saveCredentialsToKeyring()
        print('credentials migrated to keyring - you can now delete', file)


class loadui(QDialog):
    def __init__(self, path):
        super().__init__()
        loadUi(path, self)


if __name__ == '__main__':
    DB = DBconnect()
    cnx = DB.getConnection()
    if cnx is not None:
        cursor = cnx.cursor()