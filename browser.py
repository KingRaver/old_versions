#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Union, Any
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
    JavascriptException
)
from utils.logger import logger
from config import config

class BrowserSetup:
    def __init__(self) -> None:
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.chrome_driver_path: str = config.CHROME_DRIVER_PATH
        logger.logger.info(f"ChromeDriver path set to: {self.chrome_driver_path}")
        
        # Track initialization state
        self._is_initialized: bool = False

    def initialize_driver(self) -> bool:
        """Initialize Chrome WebDriver with specific settings"""
        try:
            if self._is_initialized and self.driver:
                logger.logger.info("Browser already initialized")
                return True
            
            chrome_options = Options()
        
            # Standard Chrome options
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-infobars')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-notifications')
            chrome_options.add_argument('--disable-popup-blocking')
            chrome_options.add_argument('--disable-background-timer-throttling')
            chrome_options.add_argument('--disable-backgrounding-occluded-windows')
            chrome_options.add_argument('--disable-renderer-backgrounding')
            chrome_options.add_argument('--disable-features=TranslateUI')
            chrome_options.add_argument('--disable-ipc-flooding-protection')
            chrome_options.add_argument('--disable-default-apps')
            chrome_options.add_argument('--disable-component-extensions-with-background-pages')
            chrome_options.add_argument('--disable-background-networking')
        
            # IMPORTANT: Add a custom user data directory
            import os
            user_data_dir = os.path.expanduser("~/chrome_test_profile_xtc")
            os.makedirs(user_data_dir, exist_ok=True)
            chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
        
            # Window settings
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--start-maximized')
        
            # Anti-detection options
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
        
            # Initialize the driver with modern Service approach
            service = Service(executable_path=self.chrome_driver_path)
        
            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options
            )
        
            # Set page load timeout
            self.driver.set_page_load_timeout(30)
        
            # Initialize WebDriverWait with longer timeout
            self.wait = WebDriverWait(self.driver, 20)
        
            # Execute stealth JavaScript
            self._inject_stealth_js()
        
            self._is_initialized = True
            logger.logger.info("Browser initialized successfully")
            return True
        
        except WebDriverException as e:
            logger.log_error("Browser Setup", f"Failed to initialize browser: {str(e)}", exc_info=True)
            self._is_initialized = False
            return False

    def _inject_stealth_js(self) -> None:
        """Inject JavaScript to avoid detection"""
        if not self.driver:
            logger.log_error("JavaScript Injection", "Driver not initialized")
            return
            
        stealth_js = """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        
        // Add custom user agent if needed
        Object.defineProperty(navigator, 'userAgent', {
            get: () => 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        });
        """
        try:
            self.driver.execute_script(stealth_js)
        except JavascriptException as e:
            logger.log_error("JavaScript Injection", str(e))

    def js_click(self, 
                element_identifier: str, 
                by: str = By.CSS_SELECTOR, 
                timeout: int = 10) -> bool:
        """Click element using JavaScript"""
        if not self.driver:
            logger.log_error("JavaScript Click", "Driver not initialized")
            return False
            
        try:
            element = self.wait_for_element(element_identifier, by, timeout)
            if element:
                js_click_script = """
                function simulateClick(element) {
                    // Ensure element is visible
                    element.style.opacity = '1';
                    element.style.display = 'block';
                    element.disabled = false;

                    // Simulate a complete click sequence
                    const events = ['mousedown', 'mouseup', 'click'];
                    events.forEach(eventType => {
                        const event = new MouseEvent(eventType, {
                            view: window,
                            bubbles: true,
                            cancelable: true,
                            buttons: 1
                        });
                        element.dispatchEvent(event);
                    });
                }
                arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});
                setTimeout(function() {
                    simulateClick(arguments[0]);
                }, 100);
                """
                self.driver.execute_script(js_click_script, element)
                time.sleep(1)  # Wait for click to register
                return True
            else:
                # Add explicit return for when element is None
                return False
        except Exception as e:
            logger.log_error("JavaScript Click", f"Failed to click element {element_identifier}: {str(e)}")
            return False

    def js_click_next_button(self) -> bool:
        """Specific function for Twitter next button"""
        if not self.driver:
            logger.log_error("Next Button Click", "Driver not initialized")
            return False
            
        try:
            next_button_js = """
            function findNextButton() {
                // Try multiple methods to find the button
                const selectors = [
                    '[data-testid="auth_input_forward_button"]',
                    '[role="button"]:not([data-testid="AppTabBar_More_Menu"])',
                    'div[role="button"]',
                    'div[tabindex="0"][role="button"]'
                ];
                
                for (let selector of selectors) {
                    const elements = document.querySelectorAll(selector);
                    for (let element of elements) {
                        const text = element.textContent.toLowerCase().trim();
                        const isNextButton = text === 'next' || 
                                          text === 'siguiente' || 
                                          element.getAttribute('data-testid') === 'auth_input_forward_button';
                        if (isNextButton) {
                            return element;
                        }
                    }
                }
                return null;
            }

            const button = findNextButton();
            if (button) {
                button.style.opacity = '1';
                button.style.display = 'block';
                button.disabled = false;
                
                ['mousedown', 'mouseup', 'click'].forEach(eventType => {
                    const event = new MouseEvent(eventType, {
                        view: window,
                        bubbles: true,
                        cancelable: true,
                        buttons: 1
                    });
                    button.dispatchEvent(event);
                });
                
                return true;
            }
            return false;
            """
            result = bool(self.driver.execute_script(next_button_js))
            if not result:
                logger.log_error("Next Button Click", "Next button not found")
            return result
            
        except Exception as e:
            logger.log_error("Next Button Click", str(e))
            return False

    def js_send_keys(self, 
                    element_identifier: str, 
                    text: str, 
                    by: str = By.CSS_SELECTOR, 
                    timeout: int = 10) -> bool:
        """Send keys using JavaScript with enhanced input simulation"""
        if not self.driver:
            logger.log_error("JavaScript Input", "Driver not initialized")
            return False
            
        try:
            element = self.wait_for_element(element_identifier, by, timeout)
            if element is not None:  # More explicit None check
                js_input_script = """
                function simulateInput(element, text) {
                    element.focus();
                    element.value = '';
                    element.value = arguments[1];
                    
                    const events = ['input', 'change', 'keydown', 'keyup', 'keypress'];
                    events.forEach(eventType => {
                        const event = new Event(eventType, { bubbles: true });
                        element.dispatchEvent(event);
                    });
                    
                    text.split('').forEach(char => {
                        const keyEvent = new KeyboardEvent('keypress', {
                            key: char,
                            code: 'Key' + char.toUpperCase(),
                            bubbles: true
                        });
                        element.dispatchEvent(keyEvent);
                    });
                }
                
                arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});
                setTimeout(function() {
                    simulateInput(arguments[0], arguments[1]);
                }, 100);
                """
                self.driver.execute_script(js_input_script, element, text)
                time.sleep(1)
                
                actual_value = self.driver.execute_script("return arguments[0].value;", element)
                if actual_value != text:
                    logger.log_error("JavaScript Input", f"Input verification failed for {element_identifier}")
                return actual_value == text
            
            # Element not found
            logger.log_error("JavaScript Input", f"Element not found: {element_identifier}")
            return False
            
        except Exception as e:
            logger.log_error("JavaScript Input", f"Failed to input text to {element_identifier}: {str(e)}")
            return False

    def safe_click(self, 
                   element_identifier: str, 
                   by: str = By.CSS_SELECTOR, 
                   timeout: int = 10) -> bool:
        """Try multiple click methods"""
        try:
            if self.js_click(element_identifier, by, timeout):
                return True
                
            element = self.wait_for_element(element_identifier, by, timeout)
            if element and element.is_displayed():
                element.click()
                return True
                
            logger.log_error("Click Action", f"Failed to click element {element_identifier}")
            return False
        except Exception as e:
            logger.log_error("Click Action", f"Exception while clicking {element_identifier}: {str(e)}")
            return False

    def safe_send_keys(self, 
                      element_identifier: str, 
                      text: str, 
                      by: str = By.CSS_SELECTOR, 
                      timeout: int = 10) -> bool:
        """Try multiple input methods"""
        try:
            if self.js_send_keys(element_identifier, text, by, timeout):
                return True
                
            element = self.wait_for_element(element_identifier, by, timeout)
            if element and element.is_displayed():
                element.clear()
                element.send_keys(text)
                return True
                
            logger.log_error("Input Action", f"Failed to input text to {element_identifier}")
            return False
        except Exception as e:
            logger.log_error("Input Action", f"Exception while inputting to {element_identifier}: {str(e)}")
            return False

    def wait_for_element(self, 
                        element_identifier: str, 
                        by: str = By.CSS_SELECTOR, 
                        timeout: int = 10) -> Optional[WebElement]:
        """Enhanced element wait with JavaScript check"""
        if not self.driver or not self.wait:
            logger.log_error("Element Wait", "Driver or wait not initialized")
            return None
            
        try:
            element = self.wait.until(
                EC.presence_of_element_located((by, element_identifier))
            )
            
            is_interactive = self.driver.execute_script("""
                var element = arguments[0];
                var rect = element.getBoundingClientRect();
                return (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    !!(element.offsetWidth || element.offsetHeight || element.getClientRects().length)
                );
            """, element)
            
            if not is_interactive:
                logger.log_error("Element Wait", f"Element {element_identifier} found but not interactive")
            return element if is_interactive else None
            
        except TimeoutException:
            logger.log_error("Element Wait", f"Element not found or not interactive: {element_identifier}")
            return None

    def check_element_exists(self, 
                           element_identifier: str, 
                           by: str = By.CSS_SELECTOR) -> bool:
        """Check element existence with JavaScript validation"""
        if not self.driver:
            logger.log_error("Element Check", "Driver not initialized")
            return False
            
        try:
            element = self.driver.find_element(by, element_identifier)
            is_visible = self.driver.execute_script("""
                var elem = arguments[0];
                return !!(elem.offsetWidth || elem.offsetHeight || elem.getClientRects().length);
            """, element)
            if not is_visible:
                logger.log_error("Element Check", f"Element {element_identifier} exists but not visible")
            return bool(is_visible)
        except NoSuchElementException:
            return False

    def wait_and_refresh(self, timeout: int = 5) -> None:
        """Enhanced page refresh with state check"""
        if not self.driver or not self.wait:
            logger.log_error("Page Refresh", "Driver or wait not initialized")
            return
            
        try:
            time.sleep(timeout)
            self.driver.execute_script("window.location.reload(true);")
            self.wait.until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            time.sleep(2)
        except Exception as e:
            logger.log_error("Page Refresh", str(e))

    def clear_cookies(self) -> None:
        """Clear cookies using JavaScript"""
        if not self.driver:
            logger.log_error("Cookie Clear", "Driver not initialized")
            return
            
        try:
            self.driver.execute_script("""
                var cookies = document.cookie.split(";");
                for (var i = 0; i < cookies.length; i++) {
                    var cookie = cookies[i];
                    var eqPos = cookie.indexOf("=");
                    var name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
                    document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
                }
            """)
            logger.logger.info("Cookies cleared successfully")
        except Exception as e:
            logger.log_error("Cookie Clear", str(e))

    def close_browser(self) -> None:
        """Enhanced browser cleanup"""
        try:
            if self.driver:
                self.driver.execute_script("window.onbeforeunload = null;")
                self.driver.quit()
                self._is_initialized = False
                logger.logger.info("Browser closed successfully")
        except Exception as e:
            logger.log_error("Browser Cleanup", str(e))
            self._is_initialized = False

# Create singleton instance
browser = BrowserSetup()
