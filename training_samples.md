```java
Step 30000:
Prompt:
 public void init(OutputStream out, int bufSize)
	{
		if (com.ibm.ejs.ras.TraceComponent.isAnyTracingEnabled()&&logger.isLoggable (Level.FINE)) {  //306998.15
			logger.logp(Level.FINE, CLASS_NAME,"init", "init", out);
        }
		// make sure that we don't have anything hanging around between
		// init()s -- this is the fix for the broken pipe error being

Generated Output:
 		// called in the future.
		// this is the default case, so
```
--------------------------------------------------
```java
Step 40000:
Prompt:
 public void marshall(AdminLinkProviderForUserRequest adminLinkProviderForUserRequest, ProtocolMarshaller protocolMarshaller) {

        if (adminLinkProviderForUserRequest == null) {
            throw new SdkClientException("Invalid argument passed to marshall(...)");
        }

        try {
            protocolMarshaller.marshall(adminLinkProviderForUserRequest.getUserPoolId(), USERPOOLID_BINDING);
            protocolMarshaller.marshall(adminLinkProviderForUserRequest.getDestinationUser(), DESTINATIONUSER_BINDING);
Generated Output:
             protocolMarshaller.marshall(adminLinkProviderForUserRequest.getDestinationUser(), DESTINATIONUSER_
```
--------------------------------------------------
```java
Step 50000:
Prompt:
 void writeSpan(Span span, int sizeOfSpan, Buffer result) {
    result.writeByte(SPAN.key);
    result.writeVarint(sizeOfSpan); // length prefix
Generated Output:
     result.write(span.length());
  }
```
--------------------------------------------------
```java
Step 60000:
Prompt:
 public void householderSymmetric( int row, double gamma )
    {
        int startU = row*N;

        // compute v = -gamma*A*u
        for( int i = row+1; i < N; i++ ) {
            double totalReal = 0;
            double totalImag = 0;

            // the lower triangle is not written to so it needs to traverse upwards
            // to get the information.  Reduces the number of matrix writes need
            // improving large matrix performance
            for( int j = row+1; j < i; j++ ) {
                double realA = QT.data[(j*N+i)*2];
                double imagA = -QT.data[(j*N+i)*2+1];

                double realU = QT.data[(startU+j)*2];
                double imagU = QT.data[(startU+j)*2+1];

                totalReal += realA*realU - imagA*imagU;
                totalImag += realA*imagU + imagA*realU
Generated Output:
 
            // compute the matrix
            for( int j = row; j < N;
```
--------------------------------------------------
```java
Step 70000:
Prompt:
 public static Application getApplicationFromClassPath(final String _application,
                                                          final List<String> _classpath)
        throws InstallationException
    {
Generated Output:
         return getApplicationFromClassPath(_application, _classpath, null);
    }
```
--------------------------------------------------
```java
Step 80000:
Prompt:
 @Override
    public BSONObject create(boolean array, List<String> pathParts) {
        if (rootClass == null) {
            return array? new BasicDBList() : new BasicDBObject();
        }

        if (pathParts == null) {
            try {
                return (DBObject) rootClass.newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        String path = buildPath(pathParts);
        Class returnClass;

        if ((returnClass = getClassFromCache(path)) == null) {
            returnClass = deriveClass(path, pathParts.get(pathParts.size() - 1), array);
        }

        // At this point, we know what class to construct and the class cache is properly set

        if (DBObject.class.isAssignableFrom(returnClass)) {
            try {
                return (DBObject) returnClass.newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else if (Map.class.isAssignableFrom(returnClass))
Generated Output:
             }
        } else if (DBObject.class.isAssignableFrom(returnClass
```
--------------------------------------------------
```java
Step 90000:
Prompt:
 private ResourceMethod computeMethod(ExecutableElement ee, List<ResourceArgument> arguments) {
    final String javaDoc = processingEnv.getElementUtils().getDocComment(ee);
    final Path pathAnnotation = ee.getAnnotation(Path.class);
    final Produces producesAnnotation = ee.getAnnotation(Produces.class);
    return new ResourceMethod(
        ee.getSimpleName().toString(),
        computeRequestMethod(ee),
        (pathAnnotation == null)? null : pathAnnotation.value(),
        (producesAnnotation == null)? null : Joiner.on(",").join(producesAnnotation.value()),
Generated Output:
         (javaDoc == null)? null : javaDoc.substring(1),
        (
```
--------------------------------------------------
```java
Step 100000:
Prompt:
 public static String termsStoreName(TableDefinition tableDef) {

        String objStoreName = Utils.truncateTo(objectsStoreName(tableDef), MAX_CF_NAME_LENGTH - "_Terms".length());

Generated Output:
         return objStoreName;

    }
```
--------------------------------------------------
```java
Step 110000:
Prompt:
 public static Bitmap decodeStream(InputStream stream, boolean closeStream) {
        return AsyncBitmapTexture.decodeStream(stream,
                AsyncBitmapTexture.glMaxTextureSize,
Generated Output:
                 closeStream);
    }
```
--------------------------------------------------
```java
Step 120000:
Prompt:
 private <X, Y extends AmazonWebServiceRequest> Response<X> invoke(Request<Y> request, HttpResponseHandler<AmazonWebServiceResponse<X>> responseHandler,
            ExecutionContext executionContext, URI cachedEndpoint, URI uriFromEndpointTrait) {

        executionContext.setCredentialsProvider(CredentialUtils.getCredentialsProvider(request.getOriginalRequest(), awsCredentialsProvider));
Generated Output:
         return invoke(request, responseHandler, executionContext, cachedEndpoint, uriFromEndpointTrait);
```
--------------------------------------------------
```java
Step 130000:
Prompt:
 @Override
  protected void flushAndSync(boolean durable) throws IOException {
    if (outputStream == null) {
      throw new IOException("Trying to use aborted output stream!");
    }
    if (doubleBuf.isFlushed()) {
      return;
    }
Generated Output:
     if (outputStream.isFlushed()) {
      outputStream.flush();
```
--------------------------------------------------
```java
Step 140000:
Prompt:
 public static Drawable getSelectableBackground(Context ctx) {
        int selectableBackgroundRes = getSelectableBackgroundRes(ctx);
Generated Output:
         return selectableBackground(selectableBackgroundRes);
    }
```
--------------------------------------------------
```java
Step 150000:
Prompt:
 public void rollback() {
        log.debug("Performing rollback");
        while (!operationExecutors.isEmpty()) {
            CompensatingTransactionOperationExecutor rollbackOperation = operationExecutors.pop();
            try {
                rollbackOperation.rollback();
            } catch (Exception e) {
                throw new TransactionSystemException(
Generated Output:
                     "Error while executing rollback operation", e);
            }
        }
    }
```
--------------------------------------------------
```java
Step 160000:
Prompt:
 public static void retry(Predicate<Integer> action, String label, int times, long delay) throws TooManyRetriesException {
        if (times < 1) {
            throw new IllegalArgumentException("Retry block must try at least 1 time");
        }

        if (delay < 1) {
            throw new IllegalArgumentException("Must have at least 1 ms delay");
        }

        label = label == null? "retry block" : label;

        int tries = 0;

        for (; times > tries; ++tries) {
            try {
                if (action.test(tries)) return;
            } catch (Exception e) {
                LOGGER.warn("Attempted to perform {}, but got exception", label, e);
            }

Generated Output:
             if (times > 0) {
                LOGGER.warn("Retry block must have at
```
--------------------------------------------------
```java
Step 170000:
Prompt:
 public ResourcePoolsBuilder withReplacing(ResourcePool resourcePool) {
    Map<ResourceType<?>, ResourcePool> newPools = new HashMap<>(resourcePools);
    newPools.put(resourcePool.getType(), resourcePool);
Generated Output:
     return new ResourcePoolsBuilder(newPools);
  }
--------------------------------------------------

Step 180000:
Prompt:
 @Override
    public RejectCertificateTransferResult rejectCertificateTransfer(RejectCertificateTransferRequest request) {
        request = beforeClientExecution(request);
Generated Output:
         return executeRejectCertificateTransfer(request);
    }
--------------------------------------------------

Step 190000:
Prompt:
 public static void setLearningRate(MultiLayerNetwork net, int layerNumber, ISchedule lrSchedule) {
        setLearningRate(net, layerNumber, Double.NaN, lrSchedule, true);
Generated Output:
     }
--------------------------------------------------

Step 200000:
Prompt:
 public Access authorize(final AuthenticationResult authenticationResult)
  {
    transition(State.INITIALIZED, State.AUTHORIZING);
    return doAuthorize(
        authenticationResult,
        AuthorizationUtils.authorizeAllResourceActions(
            authenticationResult,
            Iterables.transform(
                baseQuery.getDataSource().getNames(),
                AuthorizationUtils.DATASOURCE_READ_RA_GENERATOR
Generated Output:
             )
        )
    );
  }
--------------------------------------------------

Step 210000:
Prompt:
 @SuppressWarnings("unchecked")

	@Override

	public EList<IfcMaterialProfile> getMaterialProfiles() {

Generated Output:
 		return (EList<IfcMaterialProfile>) eGet(Ifc4Package.Literals
--------------------------------------------------

Step 220000:
Prompt:
 public void focusLost(FocusEvent _focusEvent) {
        short nFocusFlags = _focusEvent.FocusFlags;
        int nFocusChangeReason = nFocusFlags & FocusChangeReason.TAB;
        if (nFocusChangeReason == FocusChangeReason.TAB) {
            // get the window of the Window that has gained the Focus...
            // Note that the xWindow is just a representation of the controlwindow
            // but not of the control itself
Generated Output:
             // Note that the xWindow is not a valid window, so we'll just get the
--------------------------------------------------

Step 230000:
Prompt:
 public static void logComponent(final String type, final String details) {

    TransactionLogger instance = getInstance();
Generated Output:
     instance.logComponent(type, details);
  }
--------------------------------------------------

Step 240000:
Prompt:
 private int processSet(String regex, int i, StringBuilder result, UnicodeSet temp, ParsePosition pos) {
        try {
            pos.setIndex(i);
            UnicodeSet x = temp.clear().applyPattern(regex, pos, symbolTable, 0);
            x.complement().complement(); // hack to fix toPattern
            result.append(x.toPattern(false));
            i = pos.getIndex() - 1; // allow for the loop increment
            return i;
Generated Output:
         } catch (Exception e) {
            throw new RuntimeException(e);
        }
--------------------------------------------------

Step 250000:
Prompt:
 public Model enrichModel(Model model) {
        final List<BeanModel> dBeans = new ArrayList<>();
        final List<EnumModel> dEnums = new ArrayList<>();
        for (BeanModel bean : model.getBeans()) {
            final BeanModel dBean = enrichBean(bean);
            dBeans.add(dBean);
        }
        for (EnumModel enumModel : model.getEnums()) {
            final EnumModel dEnumModel = enrichEnum(enumModel);
Generated Output:
             dEnums.add(dEnumModel);
        }
        return new Model
--------------------------------------------------

Step 260000:
Prompt:
 protected static void buildSignatureValidationFilterIfNeeded(final SamlRegisteredService service,
                                                                 final List<MetadataFilter> metadataFilterList,
                                                                 final Resource metadataSignatureResource) throws Exception {
        val signatureValidationFilter = SamlUtils.buildSignatureValidationFilter(metadataSignatureResource);
Generated Output:
         LOGGER.debug("Signature validation filter {} is [{}]", signatureValidationFilter, metadataFilterList
--------------------------------------------------

Step 270000:
Prompt:
 public static CommerceCurrency fetchByUUID_G(String uuid, long groupId,
		boolean retrieveFromCache) {
Generated Output:
 		return getPersistence().fetchByUUID_G(uuid, groupId, retrieveFromCache);

--------------------------------------------------

Step 280000:
Prompt:
 @Override
	public Object getObject(final int columnIndex) throws SQLException {
		String columnLabel = getWrapped().getMetaData().getColumnLabel(columnIndex);

Generated Output:
 		return getObject(columnLabel);
	}
--------------------------------------------------

Step 290000:
Prompt:
 public void addField(String fieldGroup, final I_CmsFormField formField) {

        initializeFormFieldWidget(formField);
        m_fields.put(formField.getId(), formField);
        String modelId = formField.getModelId();
        m_fieldsByModelId.put(modelId, formField);
        formField.getLayoutData().put("group", fieldGroup);
Generated Output:
         m_fieldsByModelId.put(modelId, formField);
        m
--------------------------------------------------

Step 300000:
Prompt:
 public final void compress(ByteBuffer src, ByteBuffer dest) {
    final int cpLen = compress(src, src.position(), src.remaining(), dest, dest.position(), dest.remaining());
    src.position(src.limit());
Generated Output:
     dest.position(dest.position() + cpLen);
  }
--------------------------------------------------

Step 310000:
Prompt:
 public static DoubleMatrix logi(DoubleMatrix x) {
		/*# mapfct('Math.log') #*/
//RJPP-BEGIN------------------------------------------------------------
	   for (int i = 0; i < x.length; i++)
	      x.put(i, (double) Math.log(x.get(i)));
Generated Output:
 //RJPP-END------------------------------------------------------------
		return x;
	}
--------------------------------------------------

Step 320000:
Prompt:
 @Nullable
  public Point center() {
    // Store locally since rawCenter() is mutable
    double[] center = rawCenter();
    if (center!= null && center.length == 2) {
      return Point.fromLngLat(center[0], center[1]);
Generated Output:
     }
    return null;
  }
--------------------------------------------------

Step 330000:
Prompt:
 @Override
    public BatchDetectDominantLanguageResult batchDetectDominantLanguage(BatchDetectDominantLanguageRequest request) {
        request = beforeClientExecution(request);
Generated Output:
         return executeBatchDetectDominantLanguage(request);
    }
--------------------------------------------------

Step 340000:
Prompt:
 public int getIndex(String tokenForm) {
		// tokenForm = tokenForm.replaceAll("\\W+", "_");
		int[] index = (int[]) tokenForm2index.get(tokenForm);
		if (index == null)
Generated Output:
 			index = new int[0];
		return index[0];
	
--------------------------------------------------

Step 350000:
Prompt:
 public ModifierAdjustment withFieldModifiers(ElementMatcher<? super FieldDescription.InDefinedShape> matcher,
                                                 List<? extends ModifierContributor.ForField> modifierContributors) {
        return new ModifierAdjustment(typeAdjustments, CompoundList.of(new Adjustment<FieldDescription.InDefinedShape>(matcher,
Generated Output:
                                         new ModifierContributorModifierContributor(modifierContributors))));
    }
--------------------------------------------------

Step 360000:
Prompt:
 public byte[] fastpath(int fnId, FastpathArg[] args) throws SQLException {
    // Turn fastpath array into a parameter list.
    ParameterList params = executor.createFastpathParameters(args.length);
    for (int i = 0; i < args.length; ++i) {
      args[i].populateParameter(params, i + 1);
    }

Generated Output:
     // Convert the array to byte array.
    return (byte[]) executor.execute
--------------------------------------------------

Step 370000:
Prompt:
 protected final PrcBankStatementSave<RS> lazyGetPrcBankStatementSave(
    final Map<String, Object> pAddParam) throws Exception {
    @SuppressWarnings("unchecked")
    PrcBankStatementSave<RS> proc = (PrcBankStatementSave<RS>)
      this.processorsMap.get(PrcBankStatementSave.class.getSimpleName());
    if (proc == null) {
      proc = new PrcBankStatementSave<RS>();
      proc.setSrvOrm(getSrvOrm());
      proc.setCsvReader(getCsvReader());
Generated Output:
       this.processorsMap.put(PrcBankStatementSave.class.getSimpleName(),
--------------------------------------------------

Step 380000:
Prompt:
 public static TextComponent make(final @NonNull Consumer<Builder> consumer) {
    final Builder builder = builder();
    consumer.accept(builder);
Generated Output:
     return new TextComponent(builder.build());
  }
--------------------------------------------------

Step 390000:
Prompt:
 public Statistics columnDistinctCount(String columnName, Long ndv) {
		this.columnStats
			.computeIfAbsent(columnName, column -> new HashMap<>())
			.put(DISTINCT_COUNT, String.valueOf(ndv));
Generated Output:
 		return this;
	}
--------------------------------------------------

Step 400000:
Prompt:
 public void loadSystemProperties(String configurationKey) {
        String propertyRef = getProperty(configurationKey);
        if (propertyRef == null) {
            return;
        }

        if (propertyRef.startsWith("env:")) {
            propertyRef = propertyRef.substring(4);
            propertyRef = System.getenv(propertyRef);
        }

        if (!propertyRef.startsWith("/")) {
            propertyRef = "/" + propertyRef;
        }
        try {
            URL url = getClass().getResource(propertyRef);
            if (url == null) {
                url = new URL(propertyRef);
Generated Output:
             }
            loadSystemProperties(url);
        } catch (MalformedURLException e)
--------------------------------------------------

Step 410000:
Prompt:
 public void setPadding(float padding) {
		if (Float.isNaN(padding_right)) {
			setPadding_right(padding);
		}
		if (Float.isNaN(padding_left)) {
			setPadding_left(padding);
		}
		if (Float.isNaN(padding_top)) {
			setPadding_top(padding);
Generated Output:
 		}
		if (padding_bottom != null) {
			set
--------------------------------------------------

Step 420000:
Prompt:
 public static PropertyFile readOnlyUnion(final File... filenames) throws IOException
	{
		final PropertyFile props = new PropertyFile();

		for (final File f : filenames)
		{
			if (f!= null && f.exists())
			{
				props.merge(new PropertyFile(f));
			}
		}
Generated Output:
 
		if (props != null)
			return props;

		
--------------------------------------------------

Step 430000:
Prompt:
 public static Integer getDurationUnits(RecurringTask recurrence)
   {
      Duration duration = recurrence.getDuration();
      Integer result = null;

      if (duration!= null)
      {
         result = UNITS_MAP.get(duration.getUnits());
Generated Output:
       }

      return result;
   }
--------------------------------------------------

Step 440000:
Prompt:
 public StartExportTaskRequest withExportDataFormat(String... exportDataFormat) {
        if (this.exportDataFormat == null) {
            setExportDataFormat(new java.util.ArrayList<String>(exportDataFormat.length));
        }
        for (String ele : exportDataFormat) {
            this.exportDataFormat.add(ele);
Generated Output:
         }
        return this;
    }
--------------------------------------------------

Step 450000:
Prompt:
 public Predicate newPredicate(String id, Span<Term> span) {
	    idManager.updateCounter(AnnotationType.PREDICATE, id);
	Predicate newPredicate = new Predicate(id, span);
	annotationContainer.add(newPredicate, Layer.SRL, AnnotationType.PREDICATE);
Generated Output:
 	return newPredicate;
    }
--------------------------------------------------

Step 460000:
Prompt:
 public void processRecordInternally(Record record) {
        if (record instanceof FormatRecord) {
            FormatRecord fr = (FormatRecord) record;
            _customFormatRecords.put(Integer.valueOf(fr.getIndexCode()), fr);
        }
        if (record instanceof ExtendedFormatRecord) {
            ExtendedFormatRecord xr = (ExtendedFormatRecord) record;
Generated Output:
             _customFormatRecords.put(Integer.valueOf( xr.getIndexCode()
--------------------------------------------------
