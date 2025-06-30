# Testing Checklist - Smart Product Search Enhancement

## Pre-Deployment Testing Requirements

This checklist MUST be completed before deploying any changes to ensure existing functionality is not broken.

## 1. Test with Smart Search DISABLED (Default Behavior)

Set environment variable: `USE_SMART_PRODUCT_SEARCH=false` (or don't set it at all)

### Core Functionality Tests

- [ ] **Object Detection**
  - [ ] Upload a room image
  - [ ] Verify YOLOv8/Mask2Former detects objects correctly
  - [ ] Check detection confidence scores are displayed
  - [ ] Confirm all object classes are identified

- [ ] **Interactive UI**
  - [ ] Hover over detected objects - tooltip appears
  - [ ] Object highlights on hover with red outline
  - [ ] Click to select objects - selection persists
  - [ ] Click again to deselect - selection removed
  - [ ] Multiple objects can be selected
  - [ ] "Select All" button works
  - [ ] "Clear All" button works
  - [ ] Custom box drawing mode works
  - [ ] Canvas scaling works on window resize

- [ ] **Product Search (Existing)**
  - [ ] "Find Matching Products" button appears after selection
  - [ ] Text search executes successfully
  - [ ] If reverse image search is enabled, visual search works
  - [ ] Products are displayed correctly
  - [ ] Product links are valid
  - [ ] Cache is working (check second search is faster)

### Edge Cases

- [ ] Upload image with no detectable objects
- [ ] Upload corrupted/invalid image file
- [ ] Select objects then deselect all
- [ ] Test with very large image (>5MB)
- [ ] Test with very small image (<100KB)

## 2. Test with Smart Search ENABLED

Set environment variable: `USE_SMART_PRODUCT_SEARCH=true`

### Smart Search Functionality

- [ ] **Integration**
  - [ ] Smart search activates (check console for "Smart Search is ENABLED")
  - [ ] Falls back gracefully if smart search fails
  - [ ] Original search still works if smart search errors

- [ ] **Enhanced Results**
  - [ ] Products show smart scoring
  - [ ] Results are deduplicated
  - [ ] Price-appropriate items appear first
  - [ ] Style-matched items are prioritized
  - [ ] Multiple search strategies are used

### Performance Tests

- [ ] Search completes within timeout (30s default)
- [ ] No memory leaks with multiple searches
- [ ] API rate limiting is respected

## 3. Configuration Tests

### Environment Variables

Test each configuration:

- [ ] `USE_SMART_PRODUCT_SEARCH=false` - Smart search disabled
- [ ] `USE_SMART_PRODUCT_SEARCH=true` - Smart search enabled  
- [ ] `SMART_SEARCH_TIMEOUT=10` - Timeout works
- [ ] `SMART_SEARCH_MAX_RESULTS=5` - Limits results
- [ ] `SMART_SEARCH_DEBUG=true` - Debug logs appear
- [ ] `SMART_SEARCH_PRICE_ANALYSIS=false` - Price analysis disabled
- [ ] `SMART_SEARCH_STYLE_MATCHING=false` - Style matching disabled

## 4. API Key Tests

- [ ] Missing SERP_API_KEY - Graceful fallback
- [ ] Invalid SERP_API_KEY - Error handling works
- [ ] Missing IMGBB_API_KEY - Visual search disabled gracefully

## 5. User Experience Tests

### UI Flow

- [ ] No visible changes when smart search is disabled
- [ ] No UI glitches or layout issues
- [ ] Loading indicators work properly
- [ ] Error messages are user-friendly
- [ ] No console errors in browser

### Cross-Browser Testing

- [ ] Chrome/Chromium
- [ ] Firefox
- [ ] Safari
- [ ] Edge

## 6. Regression Tests

### Critical Paths

- [ ] Complete workflow: Upload → Detect → Select → Search → View Products
- [ ] Refresh page and repeat workflow
- [ ] Use browser back button during workflow
- [ ] Multiple sequential searches

## 7. Performance Benchmarks

Record baseline metrics with smart search DISABLED:

- [ ] Image upload time: _____ seconds
- [ ] Object detection time: _____ seconds
- [ ] Product search time: _____ seconds
- [ ] Total memory usage: _____ MB

Then with smart search ENABLED:

- [ ] Image upload time: _____ seconds (should be same)
- [ ] Object detection time: _____ seconds (should be same)
- [ ] Product search time: _____ seconds (may be slightly longer)
- [ ] Total memory usage: _____ MB (should be similar)

## 8. Code Review Checklist

- [ ] No modifications to existing functions
- [ ] All new code is in new files or clearly marked sections
- [ ] Feature flags protect all new functionality
- [ ] Error handling doesn't break existing flows
- [ ] No imports that could cause conflicts
- [ ] Documentation is updated

## 9. Deployment Verification

After deployment:

- [ ] Test with feature flag OFF in production
- [ ] Monitor error rates for 24 hours
- [ ] Check performance metrics
- [ ] Verify no increase in API errors
- [ ] User feedback channels monitored

## Sign-off

- [ ] All tests passed with smart search DISABLED
- [ ] All tests passed with smart search ENABLED
- [ ] No regressions identified
- [ ] Performance acceptable
- [ ] Ready for deployment

**Tested by:** _________________  
**Date:** _________________  
**Version:** _________________

## Emergency Rollback Plan

If issues are discovered:

1. Set `USE_SMART_PRODUCT_SEARCH=false` immediately
2. Monitor for resolution of issues
3. If issues persist, rollback deployment
4. Investigate in development environment
5. Re-test thoroughly before re-deployment